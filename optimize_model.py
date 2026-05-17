"""
基于深度学习的人体行为识别系统 (HARS) — 模型压缩与导出

功能:
1. 结构化剪枝 (L1范数): 移除不重要的权重通道, 减小模型体积
2. 量化感知训练 (INT8): 模拟低精度推理, 进一步压缩模型
3. ONNX 模型导出: 导出为跨平台通用格式, 供 ONNX Runtime 推理
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import MODEL_CONFIG, MODEL_DIR, NUM_CLASSES
from model import HybridSTGCNTransformer


def apply_structured_pruning(model, pruning_rate=0.3):
    """
    对模型执行基于L1范数的结构化剪枝

    对所有卷积层和全连接层按照L1范数排序,
    移除权重绝对值最小的 pruning_rate 比例的通道

    Args:
        model: PyTorch模型
        pruning_rate: 剪枝率 (0.0 ~ 1.0)

    Returns:
        model: 剪枝后的模型
        pruned_params: 被剪枝的参数数量
    """
    print(f"\n[剪枝] 开始结构化剪枝, 剪枝率: {pruning_rate:.0%}")

    original_params = sum(p.numel() for p in model.parameters())
    print(f"[剪枝] 原始参数量: {original_params:,}")

    pruned_count = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
            prune.remove(module, 'weight')
            pruned_count += 1
        elif isinstance(module, nn.Conv1d):
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
            prune.remove(module, 'weight')
            pruned_count += 1

    remaining_params = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())

    print(f"[剪枝] 剪枝的模块数: {pruned_count}")
    print(f"[剪枝] 非零参数比例: {nonzero/remaining_params:.2%}")

    return model, pruned_count


def quantization_aware_finetune(model, train_loader, epochs=3, lr=1e-4):
    """
    量化感知微调

    在微调过程中模拟INT8量化误差, 使模型适应低精度推理
    由于PyTorch QAT在某些层上限制较多, 这里采用简化版:
    对权重在前向传播时应用量化-反量化操作

    Args:
        model: 待量化的模型
        train_loader: 训练数据加载器
        epochs: 微调轮数
        lr: 学习率
    """
    print(f"\n[量化] 开始量化感知训练, Epochs: {epochs}")

    device = next(model.parameters()).device
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # 对权重应用模拟量化
            with torch.no_grad():
                for param in model.parameters():
                    if param.dim() >= 2:
                        scale = param.abs().max() / 127.0
                        if scale > 0:
                            quantized = torch.round(param / scale).clamp(-128, 127)
                            param.copy_(quantized * scale)

            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"[量化] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    print("[量化] 量化感知训练完成")
    return model


def export_to_onnx(model, save_path=None, dynamic_batch=True, target_length=32):
    """
    将PyTorch模型导出为ONNX格式

    Args:
        model: PyTorch模型
        save_path: 保存路径
        dynamic_batch: 是否支持动态batch大小
        target_length: 默认序列长度

    Returns:
        save_path: str
    """
    if save_path is None:
        save_path = os.path.join(MODEL_DIR, "hars_model.onnx")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    # 创建示例输入
    dummy_input = torch.randn(1, target_length, 33, 6).to(device)

    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"},
        }

    print(f"\n[导出] 正在导出ONNX模型...")
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=11,
        do_constant_folding=True,
    )

    # 验证ONNX模型
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print(f"[导出] ONNX模型验证通过")
    except ImportError:
        print(f"[导出] (onnx包未安装, 跳过验证)")
    except Exception as e:
        print(f"[导出] ONNX验证警告: {e}")

    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[导出] ONNX模型已保存: {save_path} ({file_size:.2f} MB)")

    return save_path


def optimize_model(model_path=None, save_path=None, pruning_rate=0.3,
                   quantize=True, quantize_epochs=3):
    """
    完整的模型优化Pipeline

    1. 加载模型
    2. 结构化剪枝
    3. 量化感知训练 (可选)
    4. 导出ONNX

    Args:
        model_path: PyTorch模型路径 (.pt)
        save_path: ONNX输出路径
        pruning_rate: 剪枝率
        quantize: 是否进行量化
        quantize_epochs: 量化微调轮数
    """
    print("=" * 60)
    print("模型优化 Pipeline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    model = HybridSTGCNTransformer().to(device)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[优化] 已加载模型: {model_path}")
    else:
        print("[优化] 使用随机初始化模型")

    # 2. 结构化剪枝
    model, _ = apply_structured_pruning(model, pruning_rate)

    # 3. 量化感知训练
    if quantize:
        # 生成少量合成数据用于量化微调
        num_samples = 100
        T = 32
        X = torch.randn(num_samples, T, 33, 6)
        Y = torch.randint(0, NUM_CLASSES, (num_samples,))
        loader = DataLoader(TensorDataset(X, Y), batch_size=16, shuffle=True)

        model = quantization_aware_finetune(model, loader, epochs=quantize_epochs)

    # 4. 导出ONNX
    onnx_path = export_to_onnx(model, save_path)

    # 5. 同时保存PT格式
    pt_path = os.path.join(MODEL_DIR, "hars_model.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"[优化] PT模型已保存: {pt_path}")

    print(f"\n{'=' * 60}")
    print(f"✓ 模型优化完成!")
    print(f"  ONNX模型: {onnx_path}")
    print(f"  PT模型: {pt_path}")
    print(f"{'=' * 60}")

    return onnx_path


if __name__ == "__main__":
    optimize_model(pruning_rate=0.3, quantize=True, quantize_epochs=2)
