"""
基于深度学习的人体行为识别系统 (HARS) — 全监督训练脚本

使用带标签的骨架数据训练 ST-GCN + Transformer 混合模型。

功能:
1. 支持从预训练权重初始化模型
2. 学习率余弦退火调度
3. 训练/验证循环
4. 最优模型保存
5. 训练日志记录

数据格式:
- 输入: (T, V=33, C=6) 骨架序列
- 标签: int, 0-9 动作类别
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    TRAIN_CONFIG, MODEL_CONFIG, MODEL_DIR, LOG_DIR,
    ACTION_CLASSES, NUM_CLASSES, NUM_KEYPOINTS
)
from model import HybridSTGCNTransformer


# ================================================================
# 训练数据集
# ================================================================

class SkeletonDataset(Dataset):
    """
    骨架动作识别数据集

    Args:
        data_list: list of (T_i, V, C) numpy数组
        labels: list of int
        target_length: 统一序列长度
        augment: 是否进行数据增强
    """

    def __init__(self, data_list, labels, target_length=32, augment=False):
        self.data = data_list
        self.labels = labels
        self.target_length = target_length
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx].copy()  # (T, V, C)
        label = self.labels[idx]

        # 数据增强
        if self.augment:
            seq = self._augment(seq)

        # 统一长度
        seq = self._pad_or_crop(seq)

        return torch.FloatTensor(seq), torch.LongTensor([label])[0]

    def _pad_or_crop(self, seq):
        T = seq.shape[0]
        if T >= self.target_length:
            start = np.random.randint(0, T - self.target_length + 1) if self.augment else 0
            return seq[start:start + self.target_length]
        else:
            pad = np.tile(seq[-1:], (self.target_length - T, 1, 1))
            return np.concatenate([seq, pad], axis=0)

    def _augment(self, seq):
        """简单数据增强"""
        # 高斯噪声
        if np.random.random() < 0.5:
            seq = seq + np.random.normal(0, 0.005, seq.shape).astype(np.float32)
        # 随机旋转
        if np.random.random() < 0.3:
            angle = np.radians(np.random.uniform(-10, 10))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            coords_xy = seq[:, :, :2].reshape(-1, 2)
            seq[:, :, :2] = (coords_xy @ R.T).reshape(seq.shape[0], seq.shape[1], 2)
        return seq


# ================================================================
# 训练主函数
# ================================================================

def train(train_data, train_labels, val_data=None, val_labels=None,
          pretrained_path=None, config=None, save_dir=None):
    """
    全监督训练

    Args:
        train_data: list of (T_i, V, C) numpy数组
        train_labels: list of int
        val_data: 验证集数据
        val_labels: 验证集标签
        pretrained_path: 预训练权重路径 (可选)
        config: 训练配置
        save_dir: 模型保存目录

    Returns:
        best_model_path: str
    """
    cfg = config or TRAIN_CONFIG
    save_dir = save_dir or MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[训练] 设备: {device}")
    print(f"[训练] 训练集: {len(train_data)} 样本")

    # 构建数据集
    train_dataset = SkeletonDataset(
        train_data, train_labels,
        target_length=cfg["max_sequence_length"],
        augment=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = None
    if val_data is not None and val_labels is not None:
        val_dataset = SkeletonDataset(
            val_data, val_labels,
            target_length=cfg["max_sequence_length"],
            augment=False
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
        print(f"[训练] 验证集: {len(val_data)} 样本")

    # 构建模型
    model = HybridSTGCNTransformer().to(device)

    # 加载预训练权重
    if pretrained_path and os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[训练] 已加载预训练权重: {pretrained_path}")
        if missing:
            print(f"  缺失的键: {len(missing)}")
        if unexpected:
            print(f"  多余的键: {len(unexpected)}")

    print(f"[训练] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和调度器
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # 训练日志
    log = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, "hars_model.pt")

    # 训练循环
    for epoch in range(cfg["epochs"]):
        # ---- 训练 ----
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            pred = output.argmax(dim=1)
            train_correct += (pred == batch_y).sum().item()
            train_total += batch_x.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ---- 验证 ----
        val_loss = 0
        val_acc = 0
        if val_loader:
            model.eval()
            val_loss_sum = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    val_loss_sum += loss.item() * batch_x.size(0)
                    pred = output.argmax(dim=1)
                    val_correct += (pred == batch_y).sum().item()
                    val_total += batch_x.size(0)

            val_loss = val_loss_sum / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            # 保存最优模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"  ★ 最优模型已保存 (Val Acc: {val_acc:.4f})")
        else:
            # 无验证集时保存最新模型
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 记录日志
        log["train_loss"].append(train_loss)
        log["train_acc"].append(train_acc)
        log["val_loss"].append(val_loss)
        log["val_acc"].append(val_acc)
        log["lr"].append(current_lr)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[训练] Epoch {epoch+1}/{cfg['epochs']} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f}")

    # 保存训练日志
    log_path = os.path.join(LOG_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n[训练] 训练日志已保存: {log_path}")
    print(f"[训练] 最优验证准确率: {best_val_acc:.4f}")

    return best_model_path


# ================================================================
# 生成合成训练数据 (用于演示)
# ================================================================

def generate_synthetic_training_data(num_per_class=50):
    """
    生成合成训练数据用于演示

    为每种动作生成模拟骨架序列:
    - 不同动作有不同的关节运动模式
    """
    data_list = []
    labels = []

    for cls_id in range(NUM_CLASSES):
        for _ in range(num_per_class):
            T = np.random.randint(20, 50)
            seq = np.zeros((T, NUM_KEYPOINTS, 6), dtype=np.float32)

            # 基础关节位置 (模拟直立人体, 归一化坐标)
            base = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
            base[0] = [0, -0.4, 0]     # 头
            base[11] = [-0.15, -0.2, 0] # 左肩
            base[12] = [0.15, -0.2, 0]  # 右肩
            base[23] = [-0.1, 0, 0]     # 左髋
            base[24] = [0.1, 0, 0]      # 右髋
            base[25] = [-0.1, 0.2, 0]   # 左膝
            base[26] = [0.1, 0.2, 0]    # 右膝
            base[27] = [-0.1, 0.4, 0]   # 左踝
            base[28] = [0.1, 0.4, 0]    # 右踝

            for t in range(T):
                coords = base.copy()

                # 根据类别添加特定运动模式
                if cls_id == 0:  # 行走
                    coords[27, 1] += 0.05 * np.sin(2 * np.pi * t / 15)
                    coords[28, 1] -= 0.05 * np.sin(2 * np.pi * t / 15)
                elif cls_id == 1:  # 跑步
                    coords[27, 1] += 0.1 * np.sin(2 * np.pi * t / 8)
                    coords[28, 1] -= 0.1 * np.sin(2 * np.pi * t / 8)
                elif cls_id == 2:  # 跳跃
                    coords[:, 1] -= 0.15 * abs(np.sin(2 * np.pi * t / 20))
                elif cls_id == 3:  # 坐下
                    coords[25, 1] = 0.1
                    coords[26, 1] = 0.1
                    coords[23, 1] = 0.15
                    coords[24, 1] = 0.15
                elif cls_id == 4:  # 站立
                    pass  # 保持默认
                elif cls_id == 5:  # 挥手
                    coords[15, 0] = -0.3 + 0.2 * np.sin(2 * np.pi * t / 10)
                    coords[15, 1] = -0.4
                elif cls_id == 6:  # 弯腰
                    coords[0, 1] = -0.2
                    coords[11, 1] = -0.05
                    coords[12, 1] = -0.05
                elif cls_id == 7:  # 举手
                    coords[15, 1] = -0.5
                    coords[16, 1] = -0.5
                elif cls_id == 8:  # 踢腿
                    coords[27, 1] -= 0.15 * abs(np.sin(2 * np.pi * t / 15))
                    coords[27, 0] -= 0.1
                elif cls_id == 9:  # 跌倒
                    fall_progress = min(t / max(T - 1, 1), 1.0)
                    coords[:, 1] += 0.2 * fall_progress
                    coords[:, 0] += 0.15 * fall_progress

                # 添加噪声
                coords += np.random.normal(0, 0.01, coords.shape).astype(np.float32)

                seq[t, :, :3] = coords
                # 速度特征
                if t > 0:
                    seq[t, :, 3:] = seq[t, :, :3] - seq[t-1, :, :3]

            data_list.append(seq)
            labels.append(cls_id)

    return data_list, labels


if __name__ == "__main__":
    print("=" * 60)
    print("全监督训练 (合成数据演示)")
    print("=" * 60)

    # 生成合成数据
    print("\n生成合成训练数据...")
    train_data, train_labels = generate_synthetic_training_data(num_per_class=30)
    val_data, val_labels = generate_synthetic_training_data(num_per_class=10)
    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")

    # 训练 (减少轮数用于快速演示)
    train(
        train_data, train_labels,
        val_data, val_labels,
        config={
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 20,
            "max_sequence_length": 32,
            "min_sequence_length": 16,
            "lr_scheduler_step": 10,
            "lr_scheduler_gamma": 0.1,
        }
    )

    print("\n✓ 训练完成!")
