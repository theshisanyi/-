"""
基于深度学习的人体行为识别系统 (HARS) — 数据预处理模块

功能:
1. MediaPipe 姿态估计: 从视频帧中提取33个3D人体关键点
2. 坐标归一化: 以髋部中心为原点, L2归一化
3. 速度特征计算: 帧间差分得到速度向量
4. 时序自适应采样: 根据运动量动态调整序列长度
5. 完整Pipeline: 视频 → (T_adaptive, 33, 6) NumPy 数组
"""

import os
import numpy as np
import cv2
from scipy.interpolate import interp1d

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python import BaseOptions as MpBaseOptions
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[警告] mediapipe 未安装, 姿态估计功能不可用。请运行: pip install mediapipe")

from config import (
    MEDIAPIPE_CONFIG, NUM_KEYPOINTS, KEYPOINT_DIM,
    KEYPOINT_INDICES, SAMPLING_CONFIG
)
from utils import normalize_landmarks, motion_energy


# MediaPipe Pose Landmarker 模型路径
_POSE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_lite.task")


def _ensure_pose_model():
    """确保姿态模型已下载"""
    if os.path.exists(_POSE_MODEL_PATH):
        return _POSE_MODEL_PATH
    os.makedirs(os.path.dirname(_POSE_MODEL_PATH), exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    print(f"[MediaPipe] 正在下载姿态模型...")
    import urllib.request
    urllib.request.urlretrieve(url, _POSE_MODEL_PATH)
    print(f"[MediaPipe] 下载完成: {_POSE_MODEL_PATH}")
    return _POSE_MODEL_PATH


class MediaPipeEstimator:
    """
    MediaPipe 姿态估计器 (Tasks API, v0.10+)
    封装 PoseLandmarker, 逐帧提取33个3D关键点坐标
    """

    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("mediapipe 未安装, 无法使用姿态估计功能")
        model_path = _ensure_pose_model()
        
        # 修复包含中文路径导致 MediaPipe 底层 C++ 加载模型崩溃从而抛出 0xC0000409 错误的问题
        try:
            with open(model_path, "rb") as f:
                model_bytes = f.read()
            base_options = MpBaseOptions(model_asset_buffer=model_bytes)
        except Exception as e:
            print(f"[MediaPipe] 读取模型文件失败: {e}")
            base_options = MpBaseOptions(model_asset_path=model_path)

        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=MEDIAPIPE_CONFIG["min_detection_confidence"],
            min_tracking_confidence=MEDIAPIPE_CONFIG["min_tracking_confidence"],
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def estimate(self, frame):
        """
        从单帧RGB图像中提取33个关键点
        Args:
            frame: OpenCV图像 (BGR, HWC)
        Returns:
            landmarks: list of [x, y, z, visibility], 长度33
                       x, y 为归一化坐标 [0,1]; z 为深度估计
            None: 如果未检测到人体
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks = []
        for lm in result.pose_landmarks[0]:
            landmarks.append([lm.x, lm.y, lm.z, lm.visibility])

        return landmarks

    def release(self):
        """释放资源"""
        self.landmarker.close()


class DataPreprocessor:
    """
    数据预处理Pipeline

    完成从原始关键点到模型输入张量的全部转换:
    1. 坐标归一化 (公式1-2)
    2. 速度特征计算 (公式3)
    3. 时序自适应采样 (公式11-13)
    """

    def __init__(self):
        self.max_frames = SAMPLING_CONFIG["max_frames"]
        self.min_frames = SAMPLING_CONFIG["min_frames"]
        self.alpha = SAMPLING_CONFIG["alpha"]

    def normalize_skeleton(self, landmarks):
        """
        骨骼归一化: 以髋部中心为原点, L2归一化

        公式(1): p'_v = p_v - p_center  (平移)
        公式(2): p''_v = p'_v / ||p'||₂  (L2归一化)

        Args:
            landmarks: (V, 4) 原始关键点 [x, y, z, visibility]
        Returns:
            normalized: (V, 3) 归一化坐标 [x, y, z]
        """
        lm = np.array(landmarks, dtype=np.float32)[:, :3]  # 取 x, y, z
        return normalize_landmarks(lm)

    def compute_velocity(self, sequence):
        """
        计算速度特征: 帧间差分

        公式(3): v_t = p_t - p_{t-1}

        Args:
            sequence: (T, V, 3) 坐标序列
        Returns:
            velocity: (T, V, 3) 速度序列 (第一帧速度为0)
        """
        seq = np.array(sequence, dtype=np.float32)
        velocity = np.zeros_like(seq)
        if len(seq) > 1:
            velocity[1:] = seq[1:] - seq[:-1]
        return velocity

    def adaptive_temporal_sampling(self, sequence):
        """
        时序自适应采样: 根据运动量动态决定目标帧数

        公式(11): M = Σ_t Σ_v ||p_t - p_{t-1}||₂  (总运动量)
        公式(12): T_target = T_min + α * (M / M_max) * (T_max - T_min)
        公式(13): 线性插值重采样到 T_target 帧

        Args:
            sequence: (T, V, C) 输入序列
        Returns:
            resampled: (T_target, V, C) 重采样后的序列
        """
        seq = np.array(sequence, dtype=np.float32)
        T, V, C = seq.shape

        if T <= self.min_frames:
            return self._resample_sequence(seq, self.min_frames)

        # 计算运动量
        M = motion_energy(seq[:, :, :3])
        M_max = max(M, 1e-6)

        # 动态目标帧数
        ratio = min(M / (M_max * 2 + 1e-6), 1.0)  # 防止超过1
        T_target = int(self.min_frames + self.alpha * ratio * (self.max_frames - self.min_frames))
        T_target = max(self.min_frames, min(T_target, self.max_frames, T))

        return self._resample_sequence(seq, T_target)

    def _resample_sequence(self, sequence, target_length):
        """
        使用线性插值将序列重采样到目标长度
        """
        T, V, C = sequence.shape
        if T == target_length:
            return sequence

        original_indices = np.linspace(0, T - 1, T)
        target_indices = np.linspace(0, T - 1, target_length)

        resampled = np.zeros((target_length, V, C), dtype=np.float32)
        for v in range(V):
            for c in range(C):
                f = interp1d(original_indices, sequence[:, v, c], kind='linear')
                resampled[:, v, c] = f(target_indices)

        return resampled

    def process_sequence(self, landmark_sequence, target_length=None):
        """
        完整预处理Pipeline: 关键点序列 → 模型输入张量

        Args:
            landmark_sequence: list of (V, 4) 原始关键点序列
            target_length: 强制目标帧数, None则自适应采样
        Returns:
            tensor: (T, V, 6) = [归一化坐标(3) + 速度特征(3)]
        """
        if len(landmark_sequence) == 0:
            return None

        # 1. 坐标归一化
        normalized_seq = []
        for landmarks in landmark_sequence:
            norm_lm = self.normalize_skeleton(landmarks)
            normalized_seq.append(norm_lm)
        normalized_seq = np.array(normalized_seq)  # (T, V, 3)

        # 2. 时序自适应采样
        if target_length is not None:
            coord_seq = self._resample_sequence(normalized_seq, target_length)
        else:
            coord_seq = self.adaptive_temporal_sampling(normalized_seq)  # (T', V, 3)

        # 3. 速度特征计算
        velocity_seq = self.compute_velocity(coord_seq)  # (T', V, 3)

        # 4. 拼接坐标 + 速度 → (T', V, 6)
        feature_seq = np.concatenate([coord_seq, velocity_seq], axis=-1)

        return feature_seq.astype(np.float32)

    def process_single_frame(self, landmarks):
        """
        处理单帧关键点（用于实时模式，不做时序采样）
        Args:
            landmarks: (V, 4) 原始关键点
        Returns:
            normalized: (V, 3) 归一化坐标
        """
        return self.normalize_skeleton(landmarks)


class VideoProcessor:
    """
    视频处理器: 从视频文件中提取骨架序列

    用于离线视频分析和训练数据准备
    """

    def __init__(self):
        self.estimator = MediaPipeEstimator()
        self.preprocessor = DataPreprocessor()

    def process_video(self, video_path, max_frames=300, callback=None):
        """
        处理视频文件, 提取骨架序列

        Args:
            video_path: 视频文件路径
            max_frames: 最大处理帧数
            callback: 进度回调函数 callback(current_frame, total_frames)
        Returns:
            result: dict {
                "landmarks": list of (V, 4),  原始关键点序列
                "features": (T, V, 6),        预处理后的特征
                "fps": float,                 视频帧率
                "total_frames": int,           总帧数
                "processed_frames": int        成功处理的帧数
            }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        landmarks_list = []

        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            lm = self.estimator.estimate(frame)
            if lm is not None:
                landmarks_list.append(lm)

            frame_idx += 1
            if callback:
                callback(frame_idx, min(total, max_frames))

        cap.release()

        # 预处理
        features = None
        if len(landmarks_list) > 0:
            features = self.preprocessor.process_sequence(landmarks_list)

        return {
            "landmarks": landmarks_list,
            "features": features,
            "fps": fps,
            "total_frames": total,
            "processed_frames": len(landmarks_list),
        }

    def release(self):
        """释放资源"""
        self.estimator.release()
