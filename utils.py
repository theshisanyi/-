"""
基于深度学习的人体行为识别系统 (HARS) — 工具函数模块

提供角度计算、骨骼绘制、坐标归一化、预测平滑等通用工具函数。
"""

import numpy as np
import cv2
import time
from config import (
    SKELETON_CONNECTIONS, COLORS, ACTION_COLORS,
    ACTION_CLASSES, ACTION_CLASSES_EN, KEYPOINT_INDICES, NUM_KEYPOINTS
)


# ================================================================
# 几何计算
# ================================================================

def calculate_angle(p1, p2, p3):
    """
    计算三个点构成的角度 (p1-p2-p3, 以p2为顶点)
    Args:
        p1, p2, p3: 各为 (x, y) 或 (x, y, z) 的numpy数组或列表
    Returns:
        角度值 (度数), 范围 [0, 180]
    """
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def calculate_distance(p1, p2):
    """计算两点之间的欧氏距离"""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_body_height(landmarks):
    """
    估算人体从头到脚的身高 (归一化坐标下)
    """
    if landmarks is None or len(landmarks) < NUM_KEYPOINTS:
        return 0
    nose = landmarks[KEYPOINT_INDICES["nose"]]
    left_ankle = landmarks[KEYPOINT_INDICES["left_ankle"]]
    right_ankle = landmarks[KEYPOINT_INDICES["right_ankle"]]
    mid_ankle = (np.array(left_ankle[:2]) + np.array(right_ankle[:2])) / 2
    return abs(nose[1] - mid_ankle[1])


def calculate_body_center(landmarks):
    """计算身体质心 (髋部中心)"""
    left_hip = np.array(landmarks[KEYPOINT_INDICES["left_hip"]][:3])
    right_hip = np.array(landmarks[KEYPOINT_INDICES["right_hip"]][:3])
    return (left_hip + right_hip) / 2


def calculate_body_tilt(landmarks):
    """
    计算身体倾斜角度 (肩膀中点与髋部中点连线 与 垂直方向 的夹角)
    返回角度 (度数), 0度=完全直立
    """
    left_shoulder = np.array(landmarks[KEYPOINT_INDICES["left_shoulder"]][:2])
    right_shoulder = np.array(landmarks[KEYPOINT_INDICES["right_shoulder"]][:2])
    left_hip = np.array(landmarks[KEYPOINT_INDICES["left_hip"]][:2])
    right_hip = np.array(landmarks[KEYPOINT_INDICES["right_hip"]][:2])

    mid_shoulder = (left_shoulder + right_shoulder) / 2
    mid_hip = (left_hip + right_hip) / 2

    body_vec = mid_shoulder - mid_hip
    vertical = np.array([0, -1])  # 屏幕坐标系, y向下
    cos_angle = np.dot(body_vec, vertical) / (np.linalg.norm(body_vec) * np.linalg.norm(vertical) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def motion_energy(sequence):
    """
    计算骨架序列的总运动量 M
    M = Σ_t Σ_v ||p(t) - p(t-1)||₂
    Args:
        sequence: (T, V, C) 骨架序列
    Returns:
        总运动量标量值
    """
    if len(sequence) < 2:
        return 0.0
    seq = np.array(sequence)
    diff = np.diff(seq, axis=0)               # (T-1, V, C)
    per_joint_motion = np.linalg.norm(diff, axis=2)  # (T-1, V)
    return float(np.sum(per_joint_motion))


# ================================================================
# 骨骼绘制
# ================================================================

def draw_skeleton(frame, landmarks, connections=None, color=None, thickness=2):
    """
    在视频帧上绘制人体骨架
    Args:
        frame: OpenCV 图像 (BGR, HWC)
        landmarks: 33个关键点的列表, 每个点为 (x, y, z) 或 (x, y, z, visibility)
                   x, y 为归一化坐标 [0,1]
        connections: 骨骼连接关系列表
        color: 骨骼颜色 (BGR), 默认使用COLORS["skeleton"]
        thickness: 线条粗细
    """
    if landmarks is None or len(landmarks) == 0:
        return frame

    h, w = frame.shape[:2]
    connections = connections or SKELETON_CONNECTIONS
    skeleton_color = color or COLORS["skeleton"]

    # 绘制连接线
    for (i, j) in connections:
        if i < len(landmarks) and j < len(landmarks):
            x1, y1 = int(landmarks[i][0] * w), int(landmarks[i][1] * h)
            x2, y2 = int(landmarks[j][0] * w), int(landmarks[j][1] * h)
            # 检查坐标有效性
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), skeleton_color, thickness, cv2.LINE_AA)

    # 绘制关键点
    for idx, lm in enumerate(landmarks):
        x, y = int(lm[0] * w), int(lm[1] * h)
        if 0 <= x < w and 0 <= y < h:
            # 重要关键点用大圆, 其他用小圆
            important_joints = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}
            radius = 5 if idx in important_joints else 3
            kp_color = COLORS["keypoint"] if idx in important_joints else skeleton_color
            cv2.circle(frame, (x, y), radius, kp_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_action_label(frame, action_id, confidence, x=20, y=50):
    """
    在视频帧上绘制动作标签和置信度 (使用英文避免 cv2.putText 中文乱码)
    """
    h, w = frame.shape[:2]
    # 使用英文标签，因为 OpenCV 原生的 getTextSize / putText 遇到中文会变成问号
    label = ACTION_CLASSES_EN.get(action_id, "Unknown")
    color = ACTION_COLORS[action_id] if action_id < len(ACTION_COLORS) else (255, 255, 255)

    # 背景矩形
    text = f"{label} ({confidence:.0%})"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.rectangle(frame, (x - 5, y - th - 10), (x + tw + 10, y + baseline + 5),
                  (0, 0, 0), -1)
    cv2.rectangle(frame, (x - 5, y - th - 10), (x + tw + 10, y + baseline + 5),
                  color, 2, cv2.LINE_AA)

    # 文字
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

    # 置信度条
    bar_y = y + baseline + 15
    bar_width = int((w - 40) * 0.4)
    cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + 12), (40, 40, 40), -1)
    filled = int(bar_width * confidence)
    if confidence > 0.7:
        bar_color = COLORS["confidence_high"]
    elif confidence > 0.4:
        bar_color = COLORS["confidence_mid"]
    else:
        bar_color = COLORS["confidence_low"]
    cv2.rectangle(frame, (x, bar_y), (x + filled, bar_y + 12), bar_color, -1)

    return frame


def draw_fps(frame, fps):
    """在帧上绘制FPS"""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (w - 160, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, COLORS["accent"], 2, cv2.LINE_AA)
    return frame


def draw_info_panel(frame, info_dict, start_y=100):
    """
    在帧右侧绘制信息面板
    Args:
        info_dict: {"标签": "值", ...}
    """
    h, w = frame.shape[:2]
    x = w - 300
    y = start_y

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - 25), (w - 10, y + len(info_dict) * 30 + 5),
                  (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for key, value in info_dict.items():
        cv2.putText(frame, f"{key}: {value}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 220), 1, cv2.LINE_AA)
        y += 28

    return frame


# ================================================================
# 数据处理
# ================================================================

def normalize_landmarks(landmarks, center_joint=None):
    """
    对关键点坐标进行归一化:
    1. 以髋部中心为原点进行平移
    2. L2归一化
    Args:
        landmarks: (V, C) numpy数组, C >= 3
        center_joint: 中心关节索引, 默认使用髋部中心
    Returns:
        归一化后的 (V, C) numpy数组
    """
    lm = np.array(landmarks, dtype=np.float32)
    if center_joint is None:
        # 使用左右髋部中心
        center = (lm[KEYPOINT_INDICES["left_hip"]] + lm[KEYPOINT_INDICES["right_hip"]]) / 2
    else:
        center = lm[center_joint]
    lm = lm - center
    # L2归一化
    norm = np.linalg.norm(lm[:, :3])
    if norm > 1e-6:
        lm[:, :3] = lm[:, :3] / norm
    return lm


def smooth_predictions(predictions, window_size=5):
    """
    对预测结果序列进行滑动窗口投票平滑
    Args:
        predictions: 最近 N 帧的预测 [(action_id, confidence), ...]
        window_size: 平滑窗口大小
    Returns:
        (smoothed_action_id, smoothed_confidence)
    """
    if not predictions:
        return 4, 0.0  # 默认: 站立, 0置信度

    recent = predictions[-window_size:]
    action_ids = [p[0] for p in recent]
    confidences = [p[1] for p in recent]

    # 加权投票, 最近的帧权重更高
    weights = np.linspace(0.5, 1.0, len(recent))
    vote_scores = {}
    for i, (aid, conf) in enumerate(recent):
        if aid not in vote_scores:
            vote_scores[aid] = 0
        vote_scores[aid] += conf * weights[i]

    best_action = max(vote_scores, key=vote_scores.get)
    total_weight = sum(weights)
    best_confidence = vote_scores[best_action] / total_weight

    return best_action, min(best_confidence, 1.0)


# ================================================================
# FPS 计数器
# ================================================================

class FPSCounter:
    """帧率计数器"""
    def __init__(self, avg_count=30):
        self.avg_count = avg_count
        self.timestamps = []

    def tick(self):
        """记录一帧"""
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.avg_count:
            self.timestamps.pop(0)

    def get_fps(self):
        """获取当前FPS"""
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed
