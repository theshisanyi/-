"""
基于深度学习的人体行为识别系统 (HARS) — 混合模型定义

模型架构: ST-GCN + Transformer 混合模型 (HybridSTGCNTransformer)

结构:
1. 多流嵌入层 (MultiStreamEmbedding): 坐标流+速度流 → 128维
2. ST-GCN块 ×2: 自适应图卷积 + 时序卷积
3. Transformer编码器 ×3: 8头自注意力 + FFN
4. 全局平均池化 + 分类头

输入: (B, T, V=33, C=6)  坐标(3) + 速度(3)
输出: (B, num_classes=10) 概率分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from config import MODEL_CONFIG, NUM_KEYPOINTS, SKELETON_CONNECTIONS


def build_adjacency_matrix():
    """
    构建人体骨架邻接矩阵 A (33x33)
    基于 MediaPipe 的骨骼连接关系
    """
    V = NUM_KEYPOINTS
    A = np.zeros((V, V), dtype=np.float32)

    # 自连接
    for i in range(V):
        A[i, i] = 1

    # 骨骼连接
    for (i, j) in SKELETON_CONNECTIONS:
        if i < V and j < V:
            A[i, j] = 1
            A[j, i] = 1

    # 归一化: D^{-1/2} A D^{-1/2}
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return A_norm


class MultiStreamEmbedding(nn.Module):
    """
    多流嵌入层

    将坐标流(3维)和速度流(3维)分别投影到 embed_dim 维,
    拼接后再投影回 embed_dim 维.

    Input: (B, T, V, C=6)
    Output: (B, T, V, embed_dim)
    """

    def __init__(self, coord_channels=3, velocity_channels=3, embed_dim=128):
        super().__init__()
        self.coord_proj = nn.Sequential(
            nn.Linear(coord_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.velocity_proj = nn.Sequential(
            nn.Linear(velocity_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, V, C) 其中 C = coord_channels + velocity_channels
        """
        coord = x[:, :, :, :3]      # (B, T, V, 3) 坐标
        velocity = x[:, :, :, 3:]   # (B, T, V, 3) 速度

        coord_emb = self.coord_proj(coord)         # (B, T, V, d)
        velocity_emb = self.velocity_proj(velocity) # (B, T, V, d)

        fused = torch.cat([coord_emb, velocity_emb], dim=-1)  # (B, T, V, 2d)
        out = self.fusion(fused)                                # (B, T, V, d)

        return out


class AdaptiveGraphConv(nn.Module):
    """
    自适应图卷积 (公式6)

    A_adaptive = A_fixed + A_learnable + A_data
    其中:
    - A_fixed: 物理骨架邻接矩阵 (不可学习)
    - A_learnable: 可学习的邻接矩阵
    - A_data: 数据驱动的注意力矩阵 (Q·K^T)

    Output = A_adaptive · X · W
    """

    def __init__(self, in_channels, out_channels, num_vertices=33):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_vertices = num_vertices

        # 固定邻接矩阵
        A_fixed = build_adjacency_matrix()
        self.register_buffer('A_fixed', torch.from_numpy(A_fixed))

        # 可学习邻接矩阵
        self.A_learnable = nn.Parameter(torch.zeros(num_vertices, num_vertices))
        nn.init.uniform_(self.A_learnable, -0.01, 0.01)

        # 数据驱动注意力
        self.query = nn.Linear(in_channels, in_channels // 4)
        self.key = nn.Linear(in_channels, in_channels // 4)

        # 权重矩阵
        self.W = nn.Linear(in_channels, out_channels, bias=True)

        # BatchNorm
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        Args:
            x: (B, T, V, C_in)
        Returns:
            out: (B, T, V, C_out)
        """
        B, T, V, C = x.shape

        # 数据驱动注意力 A_data
        Q = self.query(x)  # (B, T, V, C//4)
        K = self.key(x)    # (B, T, V, C//4)
        # (B, T, V, V)
        A_data = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(C // 4)
        A_data = F.softmax(A_data, dim=-1)

        # 自适应邻接矩阵
        A_adaptive = self.A_fixed.unsqueeze(0).unsqueeze(0) + \
                     self.A_learnable.unsqueeze(0).unsqueeze(0) + \
                     A_data  # (B, T, V, V)

        # 图卷积: A · X
        x_graph = torch.matmul(A_adaptive, x)  # (B, T, V, C_in)

        # 线性变换
        out = self.W(x_graph)  # (B, T, V, C_out)

        # BatchNorm (reshape for BN over feature dim)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, C_out, T, V)
        out = out.view(B * self.out_channels, T * V)
        # 简化BN: 对展开的维度做BN
        out = out.view(B, self.out_channels, T, V)
        out = self.bn(out.view(B, self.out_channels, -1)).view(B, self.out_channels, T, V)
        out = out.permute(0, 2, 3, 1).contiguous()  # (B, T, V, C_out)

        return out


class TemporalConv(nn.Module):
    """
    时序卷积层

    沿时间维度执行1D卷积, 捕获局部时序模式
    kernel_size = 5 (默认)
    """

    def __init__(self, channels, kernel_size=5, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, V, C)
        Returns:
            out: (B, T, V, C)
        """
        B, T, V, C = x.shape
        # 将V维度合并到batch → 对每个关节独立做时序卷积
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, V, C, T)
        x = x.view(B * V, C, T)                  # (BV, C, T)
        x = self.conv(x)                          # (BV, C, T)
        x = self.bn(x)                            # (BV, C, T)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = x.view(B, V, C, T)                    # (B, V, C, T)
        x = x.permute(0, 3, 1, 2).contiguous()    # (B, T, V, C)
        return x


class STGCNBlock(nn.Module):
    """
    ST-GCN 块: 自适应图卷积 + 时序卷积 + 残差连接

    空间维度: 图卷积捕获关节间的结构关系
    时间维度: 1D卷积捕获时序动态
    """

    def __init__(self, in_channels, out_channels, kernel_size=5,
                 num_vertices=33, dropout=0.1):
        super().__init__()
        self.gcn = AdaptiveGraphConv(in_channels, out_channels, num_vertices)
        self.tcn = TemporalConv(out_channels, kernel_size, dropout)

        # 残差连接
        self.residual = nn.Identity() if in_channels == out_channels else \
            nn.Linear(in_channels, out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, V, C_in)
        Returns:
            out: (B, T, V, C_out)
        """
        residual = self.residual(x)

        out = self.gcn(x)
        out = self.tcn(out)
        out = out + residual
        out = self.relu(out)
        out = self.dropout(out)

        return out


class PositionalEncoding(nn.Module):
    """位置编码 (正弦余弦)"""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class HybridSTGCNTransformer(nn.Module):
    """
    ST-GCN + Transformer 混合动作识别模型

    完整架构:
    Input (B, T, V=33, C=6)
        ↓
    MultiStreamEmbedding → (B, T, V, 128)
        ↓
    ST-GCN Block ×2 → (B, T, V, 128)
        ↓
    Spatial Pooling → (B, T, 128)
        ↓
    Positional Encoding
        ↓
    Transformer Encoder ×3 (8头注意力) → (B, T, 128)
        ↓
    Global Average Pooling → (B, 128)
        ↓
    Classification Head → (B, 10)
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or MODEL_CONFIG

        self.num_keypoints = cfg["num_keypoints"]
        self.embed_dim = cfg["embed_dim"]
        self.num_classes = cfg["num_classes"]

        # 1. 多流嵌入
        self.embedding = MultiStreamEmbedding(
            coord_channels=cfg["coord_channels"],
            velocity_channels=cfg["velocity_channels"],
            embed_dim=cfg["embed_dim"],
        )

        # 2. ST-GCN 层
        self.stgcn_layers = nn.ModuleList()
        for i in range(cfg["num_stgcn_layers"]):
            self.stgcn_layers.append(
                STGCNBlock(
                    in_channels=cfg["embed_dim"],
                    out_channels=cfg["embed_dim"],
                    kernel_size=cfg["temporal_kernel_size"],
                    num_vertices=cfg["num_keypoints"],
                    dropout=cfg["dropout"],
                )
            )

        # 3. 空间池化 (关节维度)
        self.spatial_pool = nn.AdaptiveAvgPool1d(1)

        # 4. 位置编码
        self.pos_encoding = PositionalEncoding(cfg["embed_dim"])

        # 5. Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg["embed_dim"],
            nhead=cfg["num_heads"],
            dim_feedforward=cfg["ffn_dim"],
            dropout=cfg["dropout"],
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg["num_transformer_layers"],
        )

        # 6. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(64, cfg["num_classes"]),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, V, C) C=6
        Returns:
            logits: (B, num_classes)
        """
        B, T, V, C = x.shape

        # 1. 多流嵌入 → (B, T, V, d)
        x = self.embedding(x)

        # 2. ST-GCN → (B, T, V, d)
        for stgcn in self.stgcn_layers:
            x = stgcn(x)

        # 3. 空间池化: (B, T, V, d) → (B, T, d)
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, T, d, V)
        x = x.view(B * T, self.embed_dim, V)     # (BT, d, V)
        x = self.spatial_pool(x).squeeze(-1)      # (BT, d)
        x = x.view(B, T, self.embed_dim)          # (B, T, d)

        # 4. 位置编码
        x = self.pos_encoding(x)

        # 5. Transformer → (B, T, d)
        x = self.transformer(x)

        # 6. 时间池化 → (B, d)
        x = x.mean(dim=1)

        # 7. 分类
        logits = self.classifier(x)

        return logits

    def get_encoder_output(self, x):
        """
        获取编码器输出 (用于对比学习预训练)
        Returns:
            features: (B, embed_dim) 全局特征向量
        """
        B, T, V, C = x.shape

        x = self.embedding(x)
        for stgcn in self.stgcn_layers:
            x = stgcn(x)

        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(B * T, self.embed_dim, V)
        x = self.spatial_pool(x).squeeze(-1)
        x = x.view(B, T, self.embed_dim)

        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # (B, embed_dim)

        return x


if __name__ == "__main__":
    # 测试模型
    print("=" * 60)
    print("HybridSTGCNTransformer 模型测试")
    print("=" * 60)

    model = HybridSTGCNTransformer()
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟输入: batch=2, T=32帧, V=33关节, C=6特征
    dummy_input = torch.randn(2, 32, 33, 6)
    print(f"输入形状: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"输出形状: {output.shape}")  # 期望 (2, 10)

    probs = F.softmax(output, dim=-1)
    print(f"预测概率: {probs[0].detach().numpy()}")
    print(f"\n✓ 模型测试通过!")
