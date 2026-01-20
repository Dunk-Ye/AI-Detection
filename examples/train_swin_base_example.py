import os
import yaml
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset

# ===================== 基础配置与工具函数 =====================
def setup_logger(log_path):
    """简单的日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def save_checkpoint(model, optimizer, epoch, acc, save_path):
    """保存模型检查点"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': acc
    }
    torch.save(checkpoint, save_path)
    logging.info(f"模型已保存至: {save_path}")

# ===================== 数据集定义 =====================
class SimpleImageDataset(Dataset):
    """简化版图像分类数据集"""
    def __init__(self, root_dir, img_size=224, is_train=True):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.img_paths, self.labels = self._load_data()
        
        # 简单的数据预处理
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        # 训练集增加数据增强
        if is_train:
            transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))
        
        self.transform = transforms.Compose(transform_list)

    def _load_data(self):
        """加载所有图片路径和标签"""
        img_paths = []
        labels = []
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(valid_exts):
                    img_paths.append(os.path.join(cls_dir, img_name))
                    labels.append(self.class_to_idx[cls])
        return img_paths, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

# ===================== 核心训练逻辑 =====================
def main():
    # 1. 加载配置
    config_path = "./configs/swin_base_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. 初始化日志
    logger = setup_logger(config["train"]["log_path"])
    
    # 3. 设置设备
    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 4. 构建数据集和数据加载器
    logger.info("加载数据集...")
    train_dataset = SimpleImageDataset(
        root_dir=config["dataset"]["train_path"],
        img_size=config["dataset"]["img_size"],
        is_train=True
    )
    val_dataset = SimpleImageDataset(
        root_dir=config["dataset"]["val_path"],
        img_size=config["dataset"]["img_size"],
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=True if torch.cuda.is_available() else False
    )
    logger.info(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")
    logger.info(f"类别数: {len(train_dataset.classes)}, 类别列表: {train_dataset.classes}")
    
    # 5. 构建模型
    logger.info("构建 Swin-base 模型...")
    model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1 if config["model"]["pretrained"] else None)
    # 修改分类头
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=config["model"]["drop_rate"]),
        nn.Linear(in_features, config["dataset"]["num_classes"])
    )
    model = model.to(device)
    
    # 6. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
        betas=tuple(config["optimizer"]["betas"])
    )
    
    # 7. 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["train"]["epochs"] - config["train"]["warmup_epochs"]
    )
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=config["train"]["warmup_epochs"]
    )
    
    # 8. 开始训练
    best_val_acc = 0.0
    logger.info("开始训练...")
    
    for epoch in range(config["train"]["epochs"]):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计指标
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': train_loss / train_total,
                'acc': train_correct / train_total
            })
        
        # 学习率更新
        if epoch < config["train"]["warmup_epochs"]:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']} [Val]")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)
                
                pbar.set_postfix({
                    'loss': val_loss / val_total,
                    'acc': val_correct / val_total
                })
        
        # 计算本轮指标
        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total
        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total
        
        logger.info(f"Epoch {epoch+1} 总结:")
        logger.info(f"训练 - 损失: {avg_train_loss:.4f}, 准确率: {avg_train_acc:.4f}")
        logger.info(f"验证 - 损失: {avg_val_loss:.4f}, 准确率: {avg_val_acc:.4f}")
        
        # 保存最优模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_checkpoint(
                model, optimizer, epoch, avg_val_acc,
                os.path.join(config["train"]["checkpoint_save_path"], "swin_base_best.pth")
            )
        
        # 每10轮保存一次检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_val_acc,
                os.path.join(config["train"]["checkpoint_save_path"], f"swin_base_epoch_{epoch+1}.pth")
            )
    
    logger.info(f"训练完成！最优验证准确率: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
