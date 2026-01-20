# Discrete Cosine Transform implemented by torch
# 试用torch实现离散余弦变化


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def DCT_mat(size):
    i = torch.arange(size, dtype=torch.float32)
    j = torch.arange(size, dtype=torch.float32)
    grid_i, grid_j = torch.meshgrid(i, j, indexing='ij')
    
    c = torch.full((size,), math.sqrt(2.0/size), dtype=torch.float32)
    c[0] = math.sqrt(1.0/size)
    c = c.unsqueeze(1)
    
    mat = c * torch.cos((grid_j + 0.5) * math.pi * grid_i / size)
    return mat

def generate_filter(start, end, size):
    r = torch.arange(size, dtype=torch.float32)
    grid_i, grid_j = torch.meshgrid(r, r, indexing='ij')
    s = grid_i + grid_j
    mask = (s >= start) & (s <= end)
    return mask.float()

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=False, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        # Buffer 默认是 float32，但在 forward 中我们会转换
        self.register_buffer('base', generate_filter(band_start, band_end, size))
        
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
        self.norm = norm
        if norm:
            self.register_buffer('ft_num', torch.sum(self.base))

    def forward(self, x):
        # [关键修复] 确保 mask 与输入 x 的 dtype 和 device 一致
        base = self.base.to(device=x.device, dtype=x.dtype)
        
        if self.use_learnable:
            # 参数通常会自动转型，但显式转换更安全
            learnable = self.learnable.to(dtype=x.dtype)
            filt = base + norm_sigma(learnable)
        else:
            filt = base

        if self.norm:
            ft_num = self.ft_num.to(device=x.device, dtype=x.dtype)
            # 防止除零，虽然这里 sum(base) 应该不为0
            y = x * filt / (ft_num + 1e-6)
        else:
            y = x * filt
        return y

class DCT_base_Rec_Module(nn.Module):
    def __init__(self, window_size=32, stride=16, output=256, grade_N=6, level_fliter=[0]):
        super().__init__()
        
        assert output % window_size == 0
        assert len(level_fliter) > 0
        
        self.window_size = window_size
        self.grade_N = grade_N
        self.level_N = len(level_fliter)
        # N 虽然这里计算了，但在Fold逻辑中，其实只用了单个Patch
        self.N = (output // window_size) * (output // window_size)
        
        self.register_buffer('_DCT_patch', DCT_mat(window_size))
        self.register_buffer('_DCT_patch_T', DCT_mat(window_size).transpose(0, 1))
        
        self.unfold = nn.Unfold(
            kernel_size=(window_size, window_size), stride=stride
        )
        # Fold输出只有 window_size x window_size，说明只拼1个块
        self.fold0 = nn.Fold(
            output_size=(window_size, window_size), 
            kernel_size=(window_size, window_size), 
            stride=window_size
        )
        
        level_f = [
            Filter(window_size, 0, window_size * 2)
        ]
        
        self.level_filters = nn.ModuleList([level_f[i] for i in level_fliter])
        self.grade_filters = nn.ModuleList([Filter(window_size, window_size * 2. / grade_N * i, window_size * 2. / grade_N * (i+1), norm=True) for i in range(grade_N)])

    def forward(self, x):
        # 1. 维度检查与升维
        is_batch = x.ndim == 4
        if not is_batch:
            x = x.unsqueeze(0)
            
        B, C, H, W = x.shape
        window_size = self.window_size
        
        dct_patch = self._DCT_patch.to(device=x.device, dtype=x.dtype)
        dct_patch_T = self._DCT_patch_T.to(device=x.device, dtype=x.dtype)
        
        # 2. Unfold: [B, C*Win*Win, L]
        x_unfold = self.unfold(x)
        L = x_unfold.shape[2]
        
        # 3. Reshape 为 [B, L, C, W, W] 方便批量DCT
        x_reshaped = x_unfold.view(B, C, window_size, window_size, L)
        x_reshaped = x_reshaped.permute(0, 4, 1, 2, 3).contiguous() # [B, L, C, W, W]
        x_processing = x_reshaped.view(-1, window_size, window_size) # [B*L*C, W, W]
        
        # 4. DCT 变换
        x_dct = dct_patch @ x_processing @ dct_patch_T
        
        # 5. Filter处理 (Level Filters)
        y_list = []
        for i in range(self.level_N):
            x_pass = self.level_filters[i](x_dct)
            y = dct_patch_T @ x_pass @ dct_patch # IDCT
            y_list.append(y)
        
        # 拼接不同 Level 的特征
        # y: [B*L*C, W, W]
        # 还原形状 -> [B, L, C, W, W]
        x_idct_levels = [item.view(B, L, C, window_size, window_size) for item in y_list]
        
        # 在 Channel 维度拼接: [B, L, level_N*C, W, W]
        level_x_batched = torch.cat(x_idct_levels, dim=2)
        
        # Flatten distinct features for gathering: [B, L, Feature_Dim]
        # Feature_Dim = level_N * C * W * W
        features_flat = level_x_batched.view(B, L, -1)
        
        # 6. 计算 Grade (用于排序)
        # 必须显式指定 dtype 避免 FP16 下创建 FP32 tensor 报错
        grade = torch.zeros(B * L, device=x.device, dtype=x.dtype)
        w, k = 1, 2
        
        # 准备数据用于 grade filter
        x_dct_4_grade = x_dct.view(B, L, C, window_size, window_size)
        
        for i in range(self.grade_N):
            _x = torch.abs(x_dct_4_grade)
            _x = torch.log(_x + 1)
            
            _x_flat = _x.view(-1, window_size, window_size)
            _x_filt = self.grade_filters[i](_x_flat)
            _x_filt = _x_filt.view(B, L, C, window_size, window_size)
            
            # Sum over C, W, W -> [B, L]
            _x_sum = torch.sum(_x_filt, dim=[2, 3, 4]) 
            
            grade = grade.view(B, L) + w * _x_sum
            w *= k
            
        # 7. 排序
        # grade: [B, L]
        # idx: [B, L] (Ascending)
        _, idx = torch.sort(grade, dim=1)
        
        # =======================================================
        # 修正逻辑：只 Gather 特定的 4 个 Patch，而不是 N 个
        # =======================================================
        
        def gather_patches(indices_bn):
            """
            indices_bn: [B, 1] - 我们只需要取1个patch
            output: [B, 1, D]
            """
            D = features_flat.shape[2]
            # 扩展索引维度以匹配 feature 维度 [B, 1, D]
            indices_expanded = indices_bn.unsqueeze(2).expand(-1, -1, D)
            return torch.gather(features_flat, 1, indices_expanded)

        # 1. MinMin (最小) -> index 0
        idx_min = idx[:, 0:1] # Slice 保持维度 [B, 1]
        
        # 2. MaxMax (最大) -> index -1 (Last)
        idx_max = idx[:, -1:] # Slice 保持维度 [B, 1]
        
        # 3. MinMin1 (次小) -> index 1
        # 简单做个边界保护，虽然 L 肯定 > 1
        if L > 1:
            idx_min1 = idx[:, 1:2]
            idx_max1 = idx[:, -2:-1]
        else:
            idx_min1 = idx_min
            idx_max1 = idx_max

        # Gather Data: [B, 1, D]
        feat_min = gather_patches(idx_min)
        feat_max = gather_patches(idx_max)
        feat_min1 = gather_patches(idx_min1)
        feat_max1 = gather_patches(idx_max1)
        
        # 8. Fold (重建)
        # Fold Input: [B, C_out*k*k, Num_Blocks]
        # 这里的 Num_Blocks 必须是 1
        # 输入 feat_min 是 [B, 1, D]，我们需要 [B, D, 1]
        
        feat_min = feat_min.transpose(1, 2)
        feat_max = feat_max.transpose(1, 2)
        feat_min1 = feat_min1.transpose(1, 2)
        feat_max1 = feat_max1.transpose(1, 2)
        
        # 执行 Fold, 输出 [B, C_out, 32, 32]
        out_min = self.fold0(feat_min)
        out_max = self.fold0(feat_max)
        out_min1 = self.fold0(feat_min1)
        out_max1 = self.fold0(feat_max1)
        
        # 9. 恢复输出格式
        if not is_batch:
            return out_min.squeeze(0), out_max.squeeze(0), out_min1.squeeze(0), out_max1.squeeze(0)
            
        return out_min, out_max, out_min1, out_max1
