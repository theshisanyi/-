"""
基于深度学习的人体行为识别系统 (HARS) — 推理引擎模块

提供三种推理模式:
1. RuleBasedEngine: 规则分类器 (无需模型, 即时可用)
2. ONNXInferenceEngine: ONNX Runtime 深度学习推理
3. HybridEngine: 融合两者结果

支持模型热更新 (增量学习后无中断替换)
"""

import numpy as np
import os
import threading

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("[警告] onnxruntime 未安装, 深度学习推理不可用。请运行: pip install onnxruntime")

from config import (
    INFERENCE_CONFIG, MODEL_CONFIG, ACTION_CLASSES, NUM_CLASSES,
    SAMPLING_CONFIG
)
from rule_classifier import RuleBasedClassifier
from data_preprocessor import DataPreprocessor


class ONNXInferenceEngine:
    """
    ONNX Runtime 推理引擎

    加载 .onnx 模型, 对预处理后的骨架序列进行推理
    """

    def __init__(self, model_path=None):
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime 未安装")

        self.model_path = model_path or INFERENCE_CONFIG["onnx_model_path"]
        self.session = None
        self.lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        """加载ONNX模型"""
        if not os.path.exists(self.model_path):
            print(f"[警告] ONNX模型不存在: {self.model_path}")
            return False

        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # 设置线程数
            sess_options.intra_op_num_threads = 4

            # 【防崩溃补丁】直接在 Python 层将模型读取为内存 bytes！
            # 彻底杜绝底层 C++ 库（ONNXRuntime）因为中文路径/特殊字符导致无法加载的问题
            with open(self.model_path, "rb") as f:
                model_bytes = f.read()

            self.session = ort.InferenceSession(
                model_bytes,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            print(f"[推理引擎] ONNX模型加载成功 (内存加载): {self.model_path}")
            return True
        except Exception as e:
            print(f"[推理引擎] ONNX模型加载失败: {e}")
            return False

    def predict(self, features):
        """
        执行推理

        Args:
            features: (T, V=33, C=6) numpy数组, 预处理后的骨架序列
        Returns:
            action_id: int
            confidence: float
            probabilities: (num_classes,) numpy数组
        """
        if self.session is None:
            return 4, 0.0, np.ones(NUM_CLASSES) / NUM_CLASSES

        with self.lock:
            # 添加batch维度 → (1, T, V, C)
            input_data = np.expand_dims(features, axis=0).astype(np.float32)

            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: input_data})[0]  # (1, num_classes)

            # Softmax
            logits = output[0]
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / (np.sum(exp_logits) + 1e-8)

            action_id = int(np.argmax(probabilities))
            confidence = float(probabilities[action_id])

            return action_id, confidence, probabilities

    def is_loaded(self):
        """检查模型是否已加载"""
        return self.session is not None

    def hot_reload(self, new_model_path=None):
        """
        热更新模型 (增量学习后调用)
        """
        path = new_model_path or self.model_path
        print(f"[推理引擎] 执行模型热更新: {path}")
        with self.lock:
            old_session = self.session
            try:
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                with open(path, "rb") as f:
                    model_bytes = f.read()
                self.session = ort.InferenceSession(
                    model_bytes, sess_options=sess_options,
                    providers=['CPUExecutionProvider']
                )
                self.model_path = path
                print("[推理引擎] 模型热更新成功")
                return True
            except Exception as e:
                print(f"[推理引擎] 模型热更新失败: {e}")
                self.session = old_session
                return False


class RuleBasedEngine:
    """
    规则分类器推理引擎

    封装 RuleBasedClassifier, 提供与 ONNXInferenceEngine 一致的接口
    """

    def __init__(self):
        self.classifier = RuleBasedClassifier(buffer_size=30)

    def predict_from_landmarks(self, landmarks):
        """
        从原始关键点预测

        Args:
            landmarks: list of [x, y, z, visibility], 长度33
        Returns:
            action_id, confidence, probabilities
        """
        action_id, confidence, scores = self.classifier.classify(landmarks)

        probabilities = np.array([scores.get(i, 0.0) for i in range(NUM_CLASSES)])

        return action_id, confidence, probabilities

    def reset(self):
        """重置分类器缓冲区"""
        self.classifier.reset()


class HybridEngine:
    """
    混合推理引擎

    融合规则分类器和深度学习模型的结果
    rule_weight: 规则分类器权重
    dl_weight: 深度学习模型权重
    """

    def __init__(self, onnx_model_path=None, rule_weight=0.4, dl_weight=0.6):
        self.rule_engine = RuleBasedEngine()
        self.dl_engine = None
        self.preprocessor = DataPreprocessor()
        self.rule_weight = rule_weight
        self.dl_weight = dl_weight
        self.landmark_buffer = []
        self.buffer_size = INFERENCE_CONFIG["buffer_size"]

        # 尝试加载深度学习模型
        if ORT_AVAILABLE:
            try:
                self.dl_engine = ONNXInferenceEngine(onnx_model_path)
                if not self.dl_engine.is_loaded():
                    self.dl_engine = None
            except Exception:
                self.dl_engine = None

    def _get_dl_features(self):
        current_len = len(self.landmark_buffer)
        if current_len < SAMPLING_CONFIG["min_frames"]:
            return None
        return self.preprocessor.process_sequence(
            list(self.landmark_buffer), target_length=32
        )

    def predict(self, landmarks, mode="hybrid"):
        """
        统一预测接口

        Args:
            landmarks: list of [x, y, z, visibility], 长度33
            mode: "rule" / "dl" / "hybrid"
        Returns:
            action_id, confidence, probabilities, mode_used
        """
        if landmarks is None:
            return 4, 0.0, np.ones(NUM_CLASSES) / NUM_CLASSES, "none"

        # 规则分类器总是运行
        rule_id, rule_conf, rule_probs = self.rule_engine.predict_from_landmarks(landmarks)

        if mode == "rule" or (mode == "hybrid" and self.dl_engine is None):
            return rule_id, rule_conf, rule_probs, "rule"

        # 深度学习推理需要积累足够帧
        self.landmark_buffer.append(landmarks)
        if len(self.landmark_buffer) > self.buffer_size:
            self.landmark_buffer.pop(0)

        if mode == "dl":
            if len(self.landmark_buffer) >= SAMPLING_CONFIG["min_frames"] and self.dl_engine is not None:
                features = self._get_dl_features()
                if features is not None:
                    dl_id, dl_conf, dl_probs = self.dl_engine.predict(features)
                    return dl_id, dl_conf, dl_probs, "dl"
            return rule_id, rule_conf, rule_probs, "rule"  # fallback

        # hybrid 模式
        if mode == "hybrid":
            if len(self.landmark_buffer) >= SAMPLING_CONFIG["min_frames"] and self.dl_engine is not None:
                features = self._get_dl_features()
                if features is not None:
                    dl_id, dl_conf, dl_probs = self.dl_engine.predict(features)

                    # 加权融合
                    combined_probs = self.rule_weight * rule_probs + self.dl_weight * dl_probs
                    combined_probs = combined_probs / (np.sum(combined_probs) + 1e-8)
                    combined_id = int(np.argmax(combined_probs))
                    combined_conf = float(combined_probs[combined_id])

                    return combined_id, combined_conf, combined_probs, "hybrid"

            return rule_id, rule_conf, rule_probs, "rule"

        return rule_id, rule_conf, rule_probs, "rule"

    def reset(self):
        """重置所有缓冲区"""
        self.rule_engine.reset()
        self.landmark_buffer.clear()

    def hot_reload_dl_model(self, model_path=None):
        """热更新深度学习模型"""
        if self.dl_engine is not None:
            return self.dl_engine.hot_reload(model_path)
        if ORT_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                self.dl_engine = ONNXInferenceEngine(model_path)
                if self.dl_engine.is_loaded():
                    print(f"[推理引擎] 首次加载DL模型成功: {model_path}")
                    return True
            except Exception as e:
                print(f"[推理引擎] 首次加载DL模型失败: {e}")
        return False
