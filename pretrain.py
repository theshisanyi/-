"""
基于深度学习的人体行为识别系统 (HARS) — SimCLR 对比学习预训练

使用自监督对比学习框架在无标签骨架数据上预训练编码器,
提升模型在小样本场景下的泛化能力。

对比学习流程:
1. 对每个骨架样本生成两个强增强视图 (view_i, view_j)
2. 使用编码器提取特征, 投影到低维空间
3. 使用NT-Xent损失最大化正样本对相似度, 最小化负样本对相似度
4. 预训练完成后, 保存编码器权重用于下游任务初始化

增强策略:
- 随机时序裁剪
- 关节随机掩码
- 关节随机旋转
- 高斯噪声注入
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from config import (
    PRETRAIN_CONFIG, MODEL_CONFIG, MODEL_DIR, NUM_KEYPOINTS
)
from model import HybridSTGCNTransformer


# ================================================================
# 数据增强
# ================================================================

def random_temporal_crop(sequence, min_ratio=0.5):
    """
    随机时序裁剪
    Args:
        sequence: (T, V, C)
        min_ratio: 最小保留比例
    """
    T = sequence.shape[0]
    crop_len = np.random.randint(int(T * min_ratio), T + 1)
    start = np.random.randint(0, T - crop_len + 1)
    return sequence[start:start + crop_len]


def random_joint_mask(sequence, mask_ratio=0.15):
    """
    随机关节掩码: 将部分关节特征置零
    """
    seq = sequence.copy()
    V = seq.shape[1]
    num_mask = max(1, int(V * mask_ratio))
    mask_joints = np.random.choice(V, num_mask, replace=False)
    seq[:, mask_joints, :] = 0
    return seq


def random_rotation(sequence, max_angle=15):
    """
    随机旋转: 在XY平面上旋转骨架
    """
    angle = np.radians(np.random.uniform(-max_angle, max_angle))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    seq = sequence.copy()
    # 只旋转坐标部分 (前2维)
    coords_xy = seq[:, :, :2]  # (T, V, 2)
    T, V, _ = coords_xy.shape
    coords_flat = coords_xy.reshape(-1, 2)  # (T*V, 2)
    rotated = coords_flat @ R.T
    seq[:, :, :2] = rotated.reshape(T, V, 2)
    vel_xy = seq[:, :, 3:5]  # (T, V, 2)
    vel_flat = vel_xy.reshape(-1, 2)
    seq[:, :, 3:5] = (vel_flat @ R.T).reshape(T, V, 2)
    return seq


def add_gaussian_noise(sequence, std=0.01):
    """添加高斯噪声"""
    noise = np.random.normal(0, std, sequence.shape).astype(np.float32)
    return sequence + noise


def augment_skeleton(sequence):
    """
    对骨架序列应用随机增强组合
    """
    seq = sequence.copy()
    # 随机时序裁剪
    seq = random_temporal_crop(seq, min_ratio=0.6)
    # 随机关节掩码 (50%概率)
    if np.random.random() < 0.5:
        seq = random_joint_mask(seq, mask_ratio=0.15)
    # 随机旋转 (70%概率)
    if np.random.random() < 0.7:
        seq = random_rotation(seq, max_angle=15)
    # 高斯噪声 (80%概率)
    if np.random.random() < 0.8:
        seq = add_gaussian_noise(seq, std=0.01)
    return seq


# ================================================================
# 投影头
# ================================================================

class ProjectionHead(nn.Module):
    """
    投影头: 将编码器输出映射到对比学习空间
    """
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ================================================================
# NT-Xent 损失
# ================================================================

class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross-Entropy) 损失

    公式(10):
    L(i,j) = -log( exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) )

    sim(z_i, z_j) = z_i · z_j / (||z_i|| * ||z_j||)  (余弦相似度)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: (B, D) 第一个视图的投影特征
            z_j: (B, D) 第二个视图的投影特征
        Returns:
            loss: 标量
        """
        B = z_i.shape[0]

        # 拼接所有特征 → (2B, D)
        z = torch.cat([z_i, z_j], dim=0)

        # 余弦相似度矩阵 → (2B, 2B)
        z_norm = F.normalize(z, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.t()) / self.temperature

        # 构造正样本对掩码
        # 对于 z[i] (i < B), 正样本是 z[i + B]
        # 对于 z[i] (i >= B), 正样本是 z[i - B]
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)

        # 移除对角线 (自身相似度)
        mask = ~torch.eye(2 * B, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(~mask, -1e9)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


# ================================================================
# 预训练数据集
# ================================================================

class UnlabeledSkeletonDataset(Dataset):
    """
    无标签骨架数据集 (用于对比学习)

    每次取样返回两个不同增强的视图
    """

    def __init__(self, data_list, target_length=32):
        """
        Args:
            data_list: list of (T_i, V, C) numpy数组
            target_length: 统一序列长度
        """
        self.data = data_list
        self.target_length = target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]  # (T, V, C)

        # 生成两个增强视图
        view_i = augment_skeleton(seq)
        view_j = augment_skeleton(seq)

        # 统一长度
        view_i = self._pad_or_crop(view_i)
        view_j = self._pad_or_crop(view_j)

        return torch.FloatTensor(view_i), torch.FloatTensor(view_j)

    def _pad_or_crop(self, seq):
        T = seq.shape[0]
        if T >= self.target_length:
            # 随机裁剪
            start = np.random.randint(0, T - self.target_length + 1)
            return seq[start:start + self.target_length]
        else:
            # 填充
            pad = np.tile(seq[-1:], (self.target_length - T, 1, 1))
            return np.concatenate([seq, pad], axis=0)


# ================================================================
# 预训练主函数
# ================================================================

def pretrain(data_list, save_path=None, config=None):
    """
    SimCLR 对比学习预训练

    Args:
        data_list: list of (T_i, V, C) numpy数组, 无标签骨架数据
        save_path: 预训练权重保存路径
        config: 预训练配置

    Returns:
        pretrained_weights_path: str
    """
    cfg = config or PRETRAIN_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[预训练] 设备: {device}")
    print(f"[预训练] 数据量: {len(data_list)} 样本")

    # 构建数据集和加载器
    dataset = UnlabeledSkeletonDataset(data_list, target_length=32)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # 构建模型 (只用编码器部分)
    model = HybridSTGCNTransformer().to(device)
    proj_head = ProjectionHead(
        input_dim=MODEL_CONFIG["embed_dim"],
        output_dim=cfg["projection_dim"],
    ).to(device)

    criterion = NTXentLoss(temperature=cfg["temperature"])
    optimizer = optim.Adam(
        list(model.parameters()) + list(proj_head.parameters()),
        lr=cfg["learning_rate"],
    )

    # 训练循环
    model.train()
    proj_head.train()

    for epoch in range(cfg["epochs"]):
        total_loss = 0
        num_batches = 0

        for view_i, view_j in dataloader:
            view_i = view_i.to(device)  # (B, T, V, C)
            view_j = view_j.to(device)

            # 编码
            feat_i = model.get_encoder_output(view_i)  # (B, d)
            feat_j = model.get_encoder_output(view_j)

            # 投影
            z_i = proj_head(feat_i)  # (B, proj_dim)
            z_j = proj_head(feat_j)

            # NT-Xent损失
            loss = criterion(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[预训练] Epoch {epoch+1}/{cfg['epochs']}, "
                  f"Loss: {avg_loss:.4f}")

    # 保存编码器权重 (不含投影头和分类头)
    if save_path is None:
        save_path = os.path.join(MODEL_DIR, "pretrain_weights.pth")

    # 只保存编码器部分的权重
    encoder_state = {}
    for name, param in model.state_dict().items():
        if not name.startswith("classifier"):
            encoder_state[name] = param

    torch.save(encoder_state, save_path)
    print(f"[预训练] 编码器权重已保存: {save_path}")

    return save_path


# ================================================================
# 生成合成预训练数据 (用于演示)
# ================================================================

def generate_synthetic_data(num_samples=200, min_length=20, max_length=60):
    """
    生成合成骨架数据用于预训练演示

    模拟不同动作的骨架运动模式
    """
    data_list = []
    for _ in range(num_samples):
        T = np.random.randint(min_length, max_length + 1)
        # 基础骨架 + 随机运动
        base = np.random.randn(1, NUM_KEYPOINTS, 6).astype(np.float32) * 0.1
        motion = np.cumsum(np.random.randn(T, NUM_KEYPOINTS, 6) * 0.01, axis=0)
        seq = base + motion
        data_list.append(seq.astype(np.float32))
    return data_list


if __name__ == "__main__":
    print("=" * 60)
    print("SimCLR 对比学习预训练")
    print("=" * 60)

    # 生成合成数据
    print("\n生成合成数据...")
    data = generate_synthetic_data(num_samples=100)
    print(f"生成 {len(data)} 个样本")

    # 预训练
    pretrain(data, config={
        "batch_size": 16,
        "learning_rate": 3e-4,
        "epochs": 10,
        "temperature": 0.07,
        "projection_dim": 64,
    })

    print("\n[OK] 预训练完成!")
