"""
基于深度学习的人体行为识别系统 (HARS) — 生成演示用ONNX模型

生成带随机权重的ONNX模型文件, 使系统在未经实际训练的情况下
也能运行深度学习推理模式 (结果为随机分类)。

用途:
- 系统功能验证
- GUI开发调试
- 毕业答辩演示 (搭配规则分类器使用)
"""

import os
import torch
from model import HybridSTGCNTransformer
from config import MODEL_DIR


def generate_demo_model():
    """生成带随机权重的ONNX演示模型"""
    print("=" * 50)
    print("生成演示用ONNX模型")
    print("=" * 50)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 创建模型
    model = HybridSTGCNTransformer()
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {params:,}")

    # 保存 PyTorch 模型
    pt_path = os.path.join(MODEL_DIR, "hars_model.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"PT模型已保存: {pt_path}")

    # 导出 ONNX
    onnx_path = os.path.join(MODEL_DIR, "hars_model.onnx")
    dummy_input = torch.randn(1, 32, 33, 6)

    try:
        # PyTorch 2.6+ 需要指定 dynamo=False 使用旧版导出器
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:
        # 旧版 PyTorch 不支持 dynamo 参数
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX模型已保存: {onnx_path} ({file_size:.2f} MB)")

    # 验证
    try:
        import onnxruntime as ort
        import numpy as np
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        test_input = np.random.randn(1, 32, 33, 6).astype(np.float32)
        result = session.run(None, {"input": test_input})[0]
        print(f"ONNX推理验证: 输出形状 {result.shape} [OK]")
    except ImportError:
        print("(onnxruntime 未安装, 跳过验证)")
    except Exception as e:
        print(f"ONNX验证警告: {e}")

    print(f"\n[OK] 演示模型生成完成!")
    return onnx_path


if __name__ == "__main__":
    generate_demo_model()
