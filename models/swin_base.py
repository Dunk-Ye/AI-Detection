# models/swin.py
import torch
import torch.nn as nn
from torchvision import models

def build_swin_base(num_classes: int, pretrained: bool = True, drop_rate: float = 0.1):
    """
    构建 Swin-base 模型
    Args:
        num_classes: 分类类别数
        pretrained: 是否加载预训练权重
        drop_rate: Dropout 率
    Returns:
        model: 配置好的 Swin-base 模型
    """
    # 加载 torchvision 官方 Swin-base 预训练模型
    model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # 修改分类头（适配自定义类别数）
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=drop_rate),
        nn.Linear(in_features, num_classes)
    )
    
    return model

if __name__ == "__main__":
    # 测试模型构建
    model = build_swin_base(num_classes=10)
    print(f"Swin-base 模型结构:\n{model}")
    # 测试前向传播
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"输出维度: {output.shape}")  # 应输出 torch.Size([1, 10])
