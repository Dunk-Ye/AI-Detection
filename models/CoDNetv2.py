# 颜色分支实现
# https://openaccess.thecvf.com/content/CVPR2025/papers/Jia_Secret_Lies_in_Color_Enhancing_AI-Generated_Images_Detection_with_Color_CVPR_2025_paper.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class AnalyticalColorFeatureExtractor(nn.Module):
    def __init__(self, bit_depth=4, noise_sigma=0.05):
        super(AnalyticalColorFeatureExtractor, self).__init__()
        self.noise_sigma = noise_sigma
        self.levels = 2 ** bit_depth
        self.scale = self.levels - 1
        
        # 这里初始化用 float32 没关系，因为它是 buffer
        # 当你在外部调用 model.half() 时，buffer 会自动被转成 half
        steps = torch.arange(self.levels, dtype=torch.float32)
        self.register_buffer('q_values', steps / self.scale)
        
        boundaries = (steps - 0.5) / self.scale
        boundaries[0] = -10.0
        self.register_buffer('boundaries', boundaries)

    def forward(self, x):
        x_unsqueezed = x.unsqueeze(-1)
        q_values = self.q_values.view(1, 1, 1, 1, -1)
        boundaries = self.boundaries.view(1, 1, 1, 1, -1)
        
        # ================= 修改开始 =================
        # 必须显式指定 dtype=x.dtype
        # 这样如果输入是 FP16，inf_tensor 也是 FP16，后续计算就不会变成 FP32
        inf_tensor = torch.tensor([10.0], device=x.device, dtype=x.dtype).view(1, 1, 1, 1, 1)
        # ================= 修改结束 =================
        
        all_boundaries = torch.cat([boundaries, inf_tensor], dim=-1)
        
        z = (all_boundaries - x_unsqueezed) / (self.noise_sigma * 1.41421356)
        
        # erf 在 FP16 下通常是支持的，只要输入全是 FP16
        cdf = 0.5 * (1 + torch.erf(z))
        
        probs = cdf[..., 1:] - cdf[..., :-1]
        x_recovered = torch.sum(probs * q_values, dim=-1)
        
        diff_map = torch.abs(x - x_recovered)
        return diff_map


class CoDNet(nn.Module):
    def __init__(self, use_fc=False):
        super(CoDNet, self).__init__()
        
        # 1. 颜色特征提取
        self.color_extractor = AnalyticalColorFeatureExtractor()
        
        # 2. 前端双分支
        self.rgb_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.diff_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # 3. 加载 Backbone
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 修改第一层卷积以适配 32 通道输入
        self.backbone.conv1 = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        
        fc_in_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Identity()
        
        self.use_fc = use_fc
        if self.use_fc:
            self.classifier = nn.Linear(fc_in_features, 2)

    def forward(self, x, use_fc=True):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            use_fc (bool): 
                - True: 返回分类结果 [B, 2] (Logits)
                - False: 返回特征向量 [B, 512] (Embeddings)
        """

        diff_map = self.color_extractor(x)
        
        feat_rgb = self.rgb_branch(x)
        feat_diff = self.diff_branch(diff_map)
        
        fused = torch.cat([feat_rgb, feat_diff], dim=1)
        
        features = self.backbone(fused) 
        
        if self.use_fc:
            out = self.classifier(features)
            return out
        else:
            return features

if __name__ == "__main__":
    model = CoDNet(use_fc=False)
    dummy_input = torch.randn(2, 3, 256, 256)
    import time
    start_time = time.time()
    output = model(dummy_input)
    end_time = time.time()
    print(f'time: {end_time-start_time}')
    print(f"Logits Shape: {output.shape}") 
