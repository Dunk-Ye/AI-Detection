import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import clip
import open_clip
from .srm_filter_kernel import all_normalized_hpf_list
# import numpy as np
import sys
import torch.nn.functional as F
from torchvision import transforms
import math

sys.path.append('/home/web_server/antispam/project/zhoudiansong/code/AIGC/CoDNet')
sys.path.append('/home/web_server/antispam/project/zhoudiansong/code/AIGC/AIDE/data')
from data.dct import DCT_base_Rec_Module
try:
    from CoDNetv2 import CoDNet
except ImportError:
    print("Warning: CoDNetv2 not found. Make sure the path is correct.")
    CoDNet = None 

# ================= LoRA 相关类 =================
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        # 复制原始层的属性
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.weight = original_linear.weight  # 共享权重（冻结）
        self.bias = original_linear.bias
        
        # LoRA 参数
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout)

        # 低秩矩阵
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            self.reset_parameters()
        
        # 冻结原始权重
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
            result += lora_out * self.scaling
        return result

def apply_lora_to_siglip(model, r=8, alpha=16, dropout=0.05):
    target_modules = ['qkv', 'proj', 'fc1', 'fc2']
    print(f"Applying LoRA (rank={r}, alpha={alpha}) to: {target_modules}")
    
    for name, module in model.named_modules():
        should_replace = False
        for target in target_modules:
            if name.endswith(target):
                should_replace = True
                break
        
        if should_replace and isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = model.get_submodule(parent_name) if '.' in name else model
            
            lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, child_name, lora_layer)
    return model
# ===============================================


class HPF(nn.Module):
  def __init__(self):
    super(HPF, self).__init__()

    all_hpf_list_5x5 = []
    for hpf_item in all_normalized_hpf_list:
      if hpf_item.shape[0] == 3:
        hpf_item = F.pad(hpf_item, (1, 1, 1, 1), mode='constant', value=0)
      all_hpf_list_5x5.append(hpf_item)

    hpf_weight = torch.stack(all_hpf_list_5x5).view(30, 1, 5, 5).contiguous()
    self.hpf_weight = torch.nn.Parameter(hpf_weight.repeat(1, 3, 1, 1), requires_grad=False)
   
    self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
    self.hpf.weight = self.hpf_weight

  def forward(self, input):
    output = self.hpf(input)
    return output


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(120, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.): 
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x) 
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class AIDE_CDS_MERGE_RESNET_Model(nn.Module):

    def __init__(self, resnet_path, convnext_path, num_classes=7, lora_r=0, lora_alpha=16, lora_dropout=0.05):
        super(AIDE_CDS_MERGE_RESNET_Model, self).__init__()
        self.hpf = HPF()
        self.model_aide = ResNet(Bottleneck, [3, 4, 6, 3])
        self.num_classes = num_classes

        if resnet_path is not None:
            try:
                pretrained_dict = torch.load(resnet_path, map_location='cpu')
                model_aide_dict = self.model_aide.state_dict()
                for k in pretrained_dict.keys():
                    if k in model_aide_dict and pretrained_dict[k].size() == model_aide_dict[k].size():
                        model_aide_dict[k] = pretrained_dict[k]
                    else:
                        pass 
                self.model_aide.load_state_dict(model_aide_dict)
                print(f"Loaded ResNet weights from {resnet_path}")
            except Exception as e:
                print(f"Error loading ResNet weights: {e}")
        
        print("build model")
        self.siglip2_vit, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP2-256", pretrained=convnext_path
        )

        self.siglip2_vit = self.siglip2_vit.visual.trunk
        self.siglip2_vit.attn_pool = nn.Identity()
        for param in self.siglip2_vit.parameters():
            param.requires_grad = False
        
        self.use_lora = (lora_r > 0)
        if self.use_lora:
            self.siglip2_vit = apply_lora_to_siglip(self.siglip2_vit, r=lora_r, alpha=lora_alpha)
            trainable_count = 0
            for name, param in self.siglip2_vit.named_parameters():
                if 'lora_' in name or 'norm' in name:
                    param.requires_grad = True
                    trainable_count += param.numel()
            print(f"LoRA enabled. Trainable params in SigLIP: {trainable_count}")
        else:
            self.siglip2_vit.eval()

        self.target_layers = [2,5,8,11]
        self.features = {}
        
        # 注册 Hook
        self._register_hooks()
        
        self.vib_z_dim = 2048
        
        self.clip_decoder = nn.Sequential(
            nn.Linear(3072, 2048),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(2048, self.vib_z_dim) 
        )
        
        if CoDNet is not None:
            self.cod = CoDNet()
        else:
            raise ImportError("CoDNet not initialized.")
        
        self.fc = Mlp(2048 + self.vib_z_dim + 512 , 1024, self.num_classes, drop=0.3) 
        
        self.norm1 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)
        self.norm3 = nn.LayerNorm(768)
        self.norm4 = nn.LayerNorm(768)

    def _get_feature_hook(self, layer_idx: int):
        # 闭包：self 绑定的是调用此方法时的实例对象
        def hook(module, input, output):
            self.features[layer_idx] = output
        return hook

    # =========================================================================
    # [Fix] 修改了 _register_hooks，使其支持清除旧Hook，解决EMA深拷贝导致指向错误的问题
    # =========================================================================
    def _register_hooks(self):
        blocks = self.siglip2_vit.blocks

        for layer_idx in self.target_layers:
            if layer_idx < len(blocks):
                layer_module = blocks[layer_idx]
                
                # 【关键修改】先尝试清空该层已有的 forward_hooks
                # 这对于 EMA 模型至关重要，因为 deepcopy 会复制指向旧模型的 Hook
                if hasattr(layer_module, '_forward_hooks'):
                    layer_module._forward_hooks.clear()
                    
                layer_module.register_forward_hook(self._get_feature_hook(layer_idx))

    def extract_features(self, image_tensor: torch.Tensor) -> dict:
        self.features = {} # 清空当前缓存
        _ = self.siglip2_vit(image_tensor)
        return self.features

    def forward(self, x):

        x_minmin = x[:, 0] 
        x_maxmax = x[:, 1]
        x_minmin1 = x[:, 2]
        x_maxmax1 = x[:, 3]
        tokens = x[:, 4]

        x_minmin = self.hpf(x_minmin)
        x_maxmax = self.hpf(x_maxmax)
        x_minmin1 = self.hpf(x_minmin1)
        x_maxmax1 = self.hpf(x_maxmax1)

        x_cod = self.cod(tokens)
        
        clip_mean = torch.Tensor([0.5,0.5,0.5]).to(tokens).view(3, 1, 1)
        clip_std = torch.Tensor([0.5,0.5,0.5]).to(tokens).view(3, 1, 1)
        dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens).view(3, 1, 1)
        dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens).view(3, 1, 1)

        img_input = tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
        
        # 无论是否 LoRA，提取特征
        if not self.use_lora:
            with torch.no_grad():
                local_convnext_image_feats = self.extract_features(img_input)
        else:
            local_convnext_image_feats = self.extract_features(img_input)

        # 安全检查，防止 AttributeError: 'NoneType' object has no attribute 'mean'
        feat0 = local_convnext_image_feats.get(self.target_layers[0])
        feat1 = local_convnext_image_feats.get(self.target_layers[1])
        feat2 = local_convnext_image_feats.get(self.target_layers[2])
        feat3 = local_convnext_image_feats.get(self.target_layers[3])

        if feat0 is None:
            # 这是一个最后的防线，如果 Hook 仍然失效，打印错误并抛出更明确的异常
            # 通常应用了上面的 _register_hooks 修复后，这里不会触发
            raise RuntimeError("Hook extraction failed! Features not found. "
                               "If this is an EMA model, ensure _register_hooks() is called after creation.")

        local0 = self.norm1(feat0.mean(dim=1).view(tokens.size(0), -1))
        local1 = self.norm2(feat1.mean(dim=1).view(tokens.size(0), -1))
        local2 = self.norm3(feat2.mean(dim=1).view(tokens.size(0), -1))
        local3 = self.norm4(feat3.mean(dim=1).view(tokens.size(0), -1))
            
        local_feat = torch.concat((local0,local1,local2,local3), dim=-1)
        x_0 = self.clip_decoder(local_feat) 

        x_aide = torch.cat((x_minmin, x_maxmax, x_minmin1, x_maxmax1), dim=1)
        x_1 = self.model_aide(x_aide)

        x = torch.cat([x_0, x_1, x_cod], dim=1)

        output = self.fc(x) 

        return output

def AIDE_CDS_MERGE_RESNET(resnet_path, convnext_path, num_classes=7, lora_r=0, lora_alpha=16):
    model = AIDE_CDS_MERGE_RESNET_Model(
        resnet_path, convnext_path, num_classes=num_classes, 
        lora_r=lora_r, lora_alpha=lora_alpha
    )
    return model

if __name__ == '__main__':
    print("Testing Model Initialization...")
    try:
        dummy_classes = 7
        # 注意这里需要你有相应的CoDNet等依赖才能运行
        # model = AIDE_CDS_MERGE_RESNET(None, None, num_classes=dummy_classes)
        print("Done.")
    except Exception as e:
        print(f"Skipping execution test: {e}")
