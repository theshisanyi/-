"""
基于深度学习的人体行为识别系统 (HARS) — 增量学习模块

实现基于 EWC (Elastic Weight Consolidation) 的在线增量学习:
1. FeedbackBuffer: FIFO缓冲区, 存储用户反馈数据
2. EWCTrainer: 计算Fisher信息矩阵, EWC正则化训练
3. BackgroundUpdater: 后台异步微调线程

当用户反馈数据达到阈值时, 自动触发后台微调, 完成后热替换推理模型
"""

import os
import json
import time
import threading
import numpy as np
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import (
    EWC_CONFIG, MODEL_CONFIG, MODEL_DIR, FEEDBACK_DIR, NUM_CLASSES
)
from model import HybridSTGCNTransformer


class FeedbackBuffer:
    """
    反馈数据FIFO缓冲区

    存储用户纠正的样本: (骨架序列, 正确标签)
    当缓冲区满时可触发微调
    """

    def __init__(self, capacity=None):
        self.capacity = capacity or EWC_CONFIG["buffer_capacity"]
        self.buffer = deque(maxlen=self.capacity)
        self.lock = threading.Lock()
        self.save_dir = FEEDBACK_DIR
        os.makedirs(self.save_dir, exist_ok=True)

    def add(self, features, label):
        """
        添加反馈样本

        Args:
            features: (T, V, C) numpy数组, 预处理后的骨架序列
            label: int, 正确的动作标签 (0-9)
        """
        with self.lock:
            self.buffer.append({
                "features": features.copy(),
                "label": int(label),
                "timestamp": datetime.now().isoformat(),
            })

    def is_full(self):
        """检查缓冲区是否达到容量"""
        with self.lock:
            return len(self.buffer) >= self.capacity

    def get_data(self):
        """
        获取缓冲区中的所有数据

        Returns:
            features_list: list of (T, V, C) numpy数组
            labels_list: list of int
        """
        with self.lock:
            features = [item["features"] for item in self.buffer]
            labels = [item["label"] for item in self.buffer]
            return features, labels

    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()

    def size(self):
        """当前样本数"""
        with self.lock:
            return len(self.buffer)

    def save_to_disk(self):
        """将反馈数据保存到磁盘"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        features_list, labels_list = self.get_data()
        if len(features_list) == 0:
            return

        save_path = os.path.join(self.save_dir, f"feedback_{timestamp}.npz")
        # 将不同长度的序列存储为object数组
        np.savez(save_path,
                 features=np.array(features_list, dtype=object),
                 labels=np.array(labels_list))
        print(f"[增量学习] 反馈数据已保存: {save_path}")


class EWCTrainer:
    """
    EWC (Elastic Weight Consolidation) 训练器

    核心思想:
    在新数据上微调时, 通过Fisher信息矩阵约束对旧任务重要的参数,
    防止灾难性遗忘.

    损失函数 (公式15):
    L_total = L_task + λ * Σ_i F_i * (θ_i - θ*_i)²

    其中:
    - L_task: 新任务交叉熵损失
    - F_i: Fisher信息矩阵对角线 (参数重要性)
    - θ*_i: 旧任务最优参数
    - λ: EWC正则化强度
    """

    def __init__(self, model=None, ewc_lambda=None, fisher_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ewc_lambda = ewc_lambda or EWC_CONFIG["ewc_lambda"]

        # 模型
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = HybridSTGCNTransformer().to(self.device)

        # Fisher信息矩阵和旧参数
        self.fisher_dict = {}
        self.optpar_dict = {}

        if fisher_path and os.path.exists(fisher_path):
            try:
                saved = torch.load(fisher_path, map_location="cpu", weights_only=True)
                self.fisher_dict = saved.get("fisher", {})
                self.optpar_dict = saved.get("optpar", {})
                print(f"[EWC] 已加载历史Fisher矩阵: {fisher_path}")
            except Exception as e:
                print(f"[EWC] 无法加载Fisher矩阵: {e}")

    def compute_fisher(self, dataloader):
        """
        计算Fisher信息矩阵 (对角近似)

        F_i = E[ (∂log p(y|x; θ) / ∂θ_i)² ]

        使用当前数据的梯度平方的期望来近似
        """
        self.model.eval()
        fisher_dict = {}
        optpar_dict = {}

        # 初始化
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)
                optpar_dict[name] = param.data.clone()

        criterion = nn.CrossEntropyLoss()

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.model.zero_grad()
            output = self.model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2

        # 取平均
        num_samples = len(dataloader.dataset)
        for name in fisher_dict:
            fisher_dict[name] /= max(num_samples, 1)

        self.fisher_dict = fisher_dict
        self.optpar_dict = optpar_dict
        print(f"[EWC] Fisher信息矩阵计算完成, 参数组数: {len(fisher_dict)}")

    def _update_fisher(self, dataloader, ema_decay=0.9):
        self.model.eval()
        new_fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_fisher[name] = torch.zeros_like(param.data)
        criterion = nn.CrossEntropyLoss()
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            self.model.zero_grad()
            output = self.model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    new_fisher[name] += param.grad.data ** 2
        num_samples = len(dataloader.dataset)
        for name in new_fisher:
            new_fisher[name] /= max(num_samples, 1)
        if not self.optpar_dict:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.optpar_dict[name] = param.data.clone()
        for name in self.fisher_dict:
            if name in new_fisher:
                self.fisher_dict[name] = ema_decay * self.fisher_dict[name] + \
                                         (1 - ema_decay) * new_fisher[name]
            else:
                self.fisher_dict[name] = new_fisher[name]
        for name in new_fisher:
            if name not in self.fisher_dict:
                self.fisher_dict[name] = new_fisher[name]
        print(f"[EWC] Fisher矩阵EMA更新完成, decay={ema_decay}")

    def ewc_loss(self):
        """
        计算EWC正则化损失

        L_ewc = λ * Σ_i F_i * (θ_i - θ*_i)²
        """
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                optpar = self.optpar_dict[name]
                loss += (fisher * (param - optpar) ** 2).sum()
        return self.ewc_lambda * loss

    def finetune(self, features_list, labels_list, epochs=None, lr=None,
                 progress_callback=None):
        """
        使用EWC在反馈数据上微调

        Args:
            features_list: list of (T, V, C) numpy数组
            labels_list: list of int
            epochs: 微调轮数
            lr: 学习率
            progress_callback: 进度回调 callback(epoch, total_epochs, loss)

        Returns:
            success: bool
        """
        epochs = epochs or EWC_CONFIG["finetune_epochs"]
        lr = lr or EWC_CONFIG["finetune_lr"]

        if len(features_list) == 0:
            return False

        # 标准化序列长度 (取最短的长度, 或使用固定长度)
        min_len = min(f.shape[0] for f in features_list)
        target_len = max(min_len, 16)

        # 裁剪所有序列到相同长度
        padded_features = []
        for f in features_list:
            if f.shape[0] >= target_len:
                padded_features.append(f[:target_len])
            else:
                # 填充 (复制最后一帧)
                pad_len = target_len - f.shape[0]
                padding = np.tile(f[-1:], (pad_len, 1, 1))
                padded_features.append(np.concatenate([f, padding], axis=0))

        X = torch.FloatTensor(np.array(padded_features)).to(self.device)
        Y = torch.LongTensor(labels_list).to(self.device)

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)

        # 先计算Fisher信息矩阵
        if len(self.fisher_dict) == 0:
            self.compute_fisher(dataloader)

        # 微调
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()

                output = self.model(batch_x)
                task_loss = criterion(output, batch_y)
                ewc_reg = self.ewc_loss()
                loss = task_loss + ewc_reg

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if progress_callback:
                progress_callback(epoch + 1, epochs, avg_loss)

            print(f"[EWC微调] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # 更新Fisher矩阵 (EMA融合, 不覆盖旧知识)
        self._update_fisher(dataloader, ema_decay=0.9)

        return True

    def save_model(self, save_path=None):
        """保存微调后的模型"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(MODEL_DIR, f"model_finetuned_{timestamp}.pt")
        torch.save(self.model.state_dict(), save_path)
        fisher_path = save_path.replace(".pt", "_fisher.pt")
        torch.save({
            "fisher": self.fisher_dict,
            "optpar": self.optpar_dict,
        }, fisher_path)
        print(f"[EWC] 模型已保存: {save_path}")
        return save_path

    def export_onnx(self, save_path=None):
        """导出为ONNX格式"""
        if save_path is None:
            save_path = os.path.join(MODEL_DIR, "hars_model.onnx")

        self.model.eval()
        dummy = torch.randn(1, 32, 33, 6).to(self.device)

        torch.onnx.export(
            self.model, dummy, save_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 1: "time"},
                "output": {0: "batch"},
            },
            opset_version=17,
        )
        print(f"[EWC] ONNX模型已导出: {save_path}")
        return save_path


class BackgroundUpdater(threading.Thread):
    """
    后台异步模型更新线程

    当反馈缓冲区满时, 在后台线程中执行EWC微调,
    完成后通知主线程进行模型热更新
    """

    def __init__(self, feedback_buffer, on_complete=None, on_progress=None):
        """
        Args:
            feedback_buffer: FeedbackBuffer 实例
            on_complete: 完成回调 callback(success, model_path)
            on_progress: 进度回调 callback(epoch, total, loss)
        """
        super().__init__(daemon=True)
        self.feedback_buffer = feedback_buffer
        self.on_complete = on_complete
        self.on_progress = on_progress
        self.trainer = None
        self._running = False

    def run(self):
        """执行后台微调"""
        self._running = True
        print("[后台更新] 开始EWC增量学习...")

        try:
            features_list, labels_list = self.feedback_buffer.get_data()

            if len(features_list) == 0:
                if self.on_complete:
                    self.on_complete(False, None)
                return

            # 创建训练器
            fisher_path = os.path.join(MODEL_DIR, "hars_model_fisher.pt")
            self.trainer = EWCTrainer(fisher_path=fisher_path)

            # 尝试加载当前最优模型
            model_path = os.path.join(MODEL_DIR, "hars_model.pt")
            if os.path.exists(model_path):
                try:
                    self.trainer.model.load_state_dict(
                        torch.load(model_path, map_location="cpu", weights_only=True)
                    )
                    print(f"[后台更新] 已加载基础模型: {model_path}")
                except Exception as e:
                    print(f"[后台更新] 无法加载基础模型, 使用随机权重: {e}")

            # 微调
            success = self.trainer.finetune(
                features_list, labels_list,
                progress_callback=self.on_progress
            )

            if success:
                # 保存更新后的模型
                pt_path = self.trainer.save_model(
                    os.path.join(MODEL_DIR, "hars_model.pt")
                )
                onnx_path = self.trainer.export_onnx(
                    os.path.join(MODEL_DIR, "hars_model.onnx")
                )

                # 保存反馈数据
                self.feedback_buffer.save_to_disk()
                self.feedback_buffer.clear()

                if self.on_complete:
                    self.on_complete(True, onnx_path)
            else:
                if self.on_complete:
                    self.on_complete(False, None)

        except Exception as e:
            print(f"[后台更新] 微调失败: {e}")
            import traceback
            traceback.print_exc()
            if self.on_complete:
                self.on_complete(False, None)

        self._running = False

    def is_running(self):
        return self._running
