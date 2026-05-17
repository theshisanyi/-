"""
基于深度学习的人体行为识别系统 (HARS) — 规则分类器

基于 MediaPipe 33个关键点的空间几何关系和时序运动特征,
使用手工设计的规则进行10种人体行为分类。

无需训练, 开箱即用, 作为实时演示的可靠方案。
支持的行为: 行走/跑步/跳跃/坐下/站立/挥手/弯腰/举手/踢腿/跌倒
"""

import numpy as np
from collections import deque
from config import KEYPOINT_INDICES as KP, ACTION_CLASSES, NUM_CLASSES
from utils import calculate_angle, calculate_distance, calculate_body_tilt


class RuleBasedClassifier:
    """
    基于几何规则的人体行为分类器

    工作原理:
    1. 从 MediaPipe 33个关键点中提取空间几何特征(角度、距离比、轴倾斜)
    2. 维护时序缓冲区, 计算运动特征(速度、周期性、突变)
    3. 对每种行为计算匹配分数, 取最高分行为为结果

    关键点索引参考 config.KEYPOINT_INDICES
    """

    def __init__(self, buffer_size=30):
        """
        Args:
            buffer_size: 时序缓冲区大小(帧数), 用于计算运动特征
        """
        self.buffer_size = buffer_size
        self.landmark_buffer = deque(maxlen=buffer_size)
        self.center_y_buffer = deque(maxlen=buffer_size)
        self.prediction_history = deque(maxlen=10)

    def classify(self, landmarks):
        """
        对单帧关键点进行行为分类

        Args:
            landmarks: list of [x, y, z, visibility], 长度33
                       x, y 为归一化坐标 [0,1]
        Returns:
            action_id: int, 0-9
            confidence: float, 0.0-1.0
            scores: dict {action_id: score}, 所有行为的分数
        """
        if landmarks is None or len(landmarks) < 33:
            return 4, 0.5, {i: 0.0 for i in range(NUM_CLASSES)}

        lm = np.array(landmarks, dtype=np.float32)
        self.landmark_buffer.append(lm.copy())

        # 计算身体中心Y坐标
        center_y = (lm[KP["left_hip"]][1] + lm[KP["right_hip"]][1]) / 2
        self.center_y_buffer.append(center_y)

        # 提取特征
        features = self._extract_features(lm)

        # 对每种行为计算匹配分数
        scores = {}
        scores[0] = self._score_walking(features)
        scores[1] = self._score_running(features)
        scores[2] = self._score_jumping(features)
        scores[3] = self._score_sitting(features)
        scores[4] = self._score_standing(features)
        scores[5] = self._score_waving(features)
        scores[6] = self._score_bending(features)
        scores[7] = self._score_raising_hand(features)
        scores[8] = self._score_kicking(features)
        scores[9] = self._score_falling(features)

        # 归一化分数
        total = sum(scores.values()) + 1e-8
        for k in scores:
            scores[k] = scores[k] / total

        # 取最高分
        best_action = max(scores, key=scores.get)
        confidence = scores[best_action]

        return best_action, confidence, scores

    def _extract_features(self, lm):
        """
        从关键点提取几何特征和运动特征

        Returns:
            dict: 各种特征值
        """
        f = {}

        # ---- 关键关节坐标 ----
        nose = lm[KP["nose"]][:2]
        l_shoulder = lm[KP["left_shoulder"]][:2]
        r_shoulder = lm[KP["right_shoulder"]][:2]
        l_elbow = lm[KP["left_elbow"]][:2]
        r_elbow = lm[KP["right_elbow"]][:2]
        l_wrist = lm[KP["left_wrist"]][:2]
        r_wrist = lm[KP["right_wrist"]][:2]
        l_hip = lm[KP["left_hip"]][:2]
        r_hip = lm[KP["right_hip"]][:2]
        l_knee = lm[KP["left_knee"]][:2]
        r_knee = lm[KP["right_knee"]][:2]
        l_ankle = lm[KP["left_ankle"]][:2]
        r_ankle = lm[KP["right_ankle"]][:2]

        mid_shoulder = (l_shoulder + r_shoulder) / 2
        mid_hip = (l_hip + r_hip) / 2
        mid_ankle = (l_ankle + r_ankle) / 2

        # ---- 空间特征 ----

        # 膝关节角度
        f["left_knee_angle"] = calculate_angle(l_hip, l_knee, l_ankle)
        f["right_knee_angle"] = calculate_angle(r_hip, r_knee, r_ankle)
        f["avg_knee_angle"] = (f["left_knee_angle"] + f["right_knee_angle"]) / 2

        # 髋关节角度 (肩-髋-膝)
        f["left_hip_angle"] = calculate_angle(l_shoulder, l_hip, l_knee)
        f["right_hip_angle"] = calculate_angle(r_shoulder, r_hip, r_knee)
        f["avg_hip_angle"] = (f["left_hip_angle"] + f["right_hip_angle"]) / 2

        # 肘关节角度
        f["left_elbow_angle"] = calculate_angle(l_shoulder, l_elbow, l_wrist)
        f["right_elbow_angle"] = calculate_angle(r_shoulder, r_elbow, r_wrist)

        # 身体倾斜角 (与垂直方向)
        f["body_tilt"] = calculate_body_tilt(lm)

        # 手腕相对肩膀高度 (y值越小越高)
        f["left_wrist_above_shoulder"] = l_shoulder[1] - l_wrist[1]
        f["right_wrist_above_shoulder"] = r_shoulder[1] - r_wrist[1]
        f["max_wrist_above_shoulder"] = max(f["left_wrist_above_shoulder"],
                                             f["right_wrist_above_shoulder"])

        # 手腕相对头部高度
        f["left_wrist_above_head"] = nose[1] - l_wrist[1]
        f["right_wrist_above_head"] = nose[1] - r_wrist[1]
        f["max_wrist_above_head"] = max(f["left_wrist_above_head"],
                                         f["right_wrist_above_head"])

        # 身体重心高度 (归一化)
        f["center_y"] = mid_hip[1]

        # 脚踝高度差 (用于检测踢腿)
        f["ankle_height_diff"] = abs(l_ankle[1] - r_ankle[1])

        # 上半身长度 (肩到髋)
        f["upper_body_len"] = calculate_distance(mid_shoulder, mid_hip)

        # 脚踝相对髋部高度
        f["left_ankle_above_hip"] = l_hip[1] - l_ankle[1]
        f["right_ankle_above_hip"] = r_hip[1] - r_ankle[1]
        f["max_ankle_above_hip"] = max(l_hip[1] - l_ankle[1], r_hip[1] - r_ankle[1])

        # ---- 运动特征 (时序) ----
        if len(self.landmark_buffer) >= 3:
            recent = list(self.landmark_buffer)

            # 整体运动速度
            diffs = []
            for i in range(1, len(recent)):
                diff = np.linalg.norm(recent[i][:, :2] - recent[i-1][:, :2], axis=1)
                diffs.append(np.mean(diff))
            f["avg_motion_speed"] = np.mean(diffs)
            f["max_motion_speed"] = np.max(diffs) if diffs else 0

            # 脚部运动速度
            foot_speed_l = []
            foot_speed_r = []
            for i in range(1, len(recent)):
                sp_l = np.linalg.norm(recent[i][KP["left_ankle"]][:2] -
                                       recent[i-1][KP["left_ankle"]][:2])
                sp_r = np.linalg.norm(recent[i][KP["right_ankle"]][:2] -
                                       recent[i-1][KP["right_ankle"]][:2])
                foot_speed_l.append(sp_l)
                foot_speed_r.append(sp_r)
            f["avg_foot_speed"] = (np.mean(foot_speed_l) + np.mean(foot_speed_r)) / 2
            f["max_foot_speed"] = max(np.max(foot_speed_l), np.max(foot_speed_r))

            # 手部运动速度
            hand_speed_l = []
            hand_speed_r = []
            for i in range(1, len(recent)):
                sp_l = np.linalg.norm(recent[i][KP["left_wrist"]][:2] -
                                       recent[i-1][KP["left_wrist"]][:2])
                sp_r = np.linalg.norm(recent[i][KP["right_wrist"]][:2] -
                                       recent[i-1][KP["right_wrist"]][:2])
                hand_speed_l.append(sp_l)
                hand_speed_r.append(sp_r)
            f["avg_hand_speed"] = (np.mean(hand_speed_l) + np.mean(hand_speed_r)) / 2
            f["max_hand_speed"] = max(np.max(hand_speed_l), np.max(hand_speed_r))

            # 重心Y坐标变化
            if len(self.center_y_buffer) >= 5:
                cy_list = list(self.center_y_buffer)
                f["center_y_velocity"] = cy_list[-1] - cy_list[-5]  # 正值=下降, 负值=上升
                f["center_y_std"] = np.std(cy_list[-10:]) if len(cy_list) >= 10 else 0
            else:
                f["center_y_velocity"] = 0
                f["center_y_std"] = 0

            # 手腕水平运动周期性 (挥手检测)
            if len(recent) >= 10:
                wrist_x = [r[KP["left_wrist"]][0] for r in recent[-15:]]
                wrist_x2 = [r[KP["right_wrist"]][0] for r in recent[-15:]]
                f["left_wrist_x_range"] = max(wrist_x) - min(wrist_x)
                f["right_wrist_x_range"] = max(wrist_x2) - min(wrist_x2)

                # 检测方向变化次数 (周期性)
                dx = np.diff(wrist_x)
                sign_changes_l = np.sum(np.abs(np.diff(np.sign(dx))) > 0)
                dx2 = np.diff(wrist_x2)
                sign_changes_r = np.sum(np.abs(np.diff(np.sign(dx2))) > 0)
                f["wrist_oscillation"] = max(sign_changes_l, sign_changes_r)
            else:
                f["left_wrist_x_range"] = 0
                f["right_wrist_x_range"] = 0
                f["wrist_oscillation"] = 0

            # 脚步交替 (行走/跑步)
            if len(recent) >= 10:
                l_ankle_y = [r[KP["left_ankle"]][1] for r in recent[-15:]]
                r_ankle_y = [r[KP["right_ankle"]][1] for r in recent[-15:]]
                diff_ankle = np.array(l_ankle_y) - np.array(r_ankle_y)
                sign_changes_feet = np.sum(np.abs(np.diff(np.sign(diff_ankle))) > 0)
                f["foot_alternation"] = sign_changes_feet
            else:
                f["foot_alternation"] = 0

        else:
            # 缓冲区不足, 设置默认运动特征
            f["avg_motion_speed"] = 0
            f["max_motion_speed"] = 0
            f["avg_foot_speed"] = 0
            f["max_foot_speed"] = 0
            f["avg_hand_speed"] = 0
            f["max_hand_speed"] = 0
            f["center_y_velocity"] = 0
            f["center_y_std"] = 0
            f["left_wrist_x_range"] = 0
            f["right_wrist_x_range"] = 0
            f["wrist_oscillation"] = 0
            f["foot_alternation"] = 0

        return f

    # ================================================================
    # 各行为评分函数
    # ================================================================

    def _score_standing(self, f):
        """站立: 直立姿态 + 低运动"""
        score = 0.0
        # 膝关节接近伸直
        if f["avg_knee_angle"] > 155:
            score += 3.0
        elif f["avg_knee_angle"] > 140:
            score += 1.5
        # 髋关节接近伸直
        if f["avg_hip_angle"] > 155:
            score += 2.0
        elif f["avg_hip_angle"] > 140:
            score += 1.0
        # 身体直立
        if f["body_tilt"] < 15:
            score += 2.0
        elif f["body_tilt"] < 25:
            score += 1.0
        # 低运动速度
        if f["avg_motion_speed"] < 0.008:
            score += 3.0
        elif f["avg_motion_speed"] < 0.015:
            score += 1.5
        # 手低于肩
        if f["max_wrist_above_shoulder"] < 0.02:
            score += 1.0
        return score

    def _score_sitting(self, f):
        """坐下: 膝关节弯曲 + 髋角减小 + 低运动"""
        score = 0.0
        # 膝关节弯曲
        if f["avg_knee_angle"] < 120:
            score += 3.0
        elif f["avg_knee_angle"] < 140:
            score += 2.0
        elif f["avg_knee_angle"] < 155:
            score += 1.0
        # 髋角减小
        if f["avg_hip_angle"] < 120:
            score += 2.5
        elif f["avg_hip_angle"] < 140:
            score += 1.5
        # 低运动
        if f["avg_motion_speed"] < 0.01:
            score += 2.0
        elif f["avg_motion_speed"] < 0.02:
            score += 1.0
        # 身体相对直立 (坐着不一定弯腰)
        if f["body_tilt"] < 30:
            score += 1.0
        return score

    def _score_walking(self, f):
        """行走: 中等速度脚步交替 + 适度运动"""
        score = 0.0
        # 脚步交替
        if f["foot_alternation"] >= 3:
            score += 3.0
        elif f["foot_alternation"] >= 2:
            score += 2.0
        elif f["foot_alternation"] >= 1:
            score += 1.0
        # 中等脚部运动
        if 0.008 < f["avg_foot_speed"] < 0.04:
            score += 2.5
        elif 0.005 < f["avg_foot_speed"] < 0.06:
            score += 1.5
        # 总体运动适度
        if 0.008 < f["avg_motion_speed"] < 0.035:
            score += 2.0
        elif 0.005 < f["avg_motion_speed"] < 0.05:
            score += 1.0
        # 身体直立
        if f["body_tilt"] < 20:
            score += 1.5
        # 膝关节有弯曲但不太大
        if 130 < f["avg_knee_angle"] < 175:
            score += 1.0
        return score

    def _score_running(self, f):
        """跑步: 高速脚步交替 + 大幅运动"""
        score = 0.0
        # 高速脚步交替
        if f["foot_alternation"] >= 3:
            score += 2.0
        elif f["foot_alternation"] >= 2:
            score += 1.0
        # 高脚部运动速度
        if f["avg_foot_speed"] > 0.04:
            score += 3.0
        elif f["avg_foot_speed"] > 0.025:
            score += 2.0
        # 高总体运动
        if f["avg_motion_speed"] > 0.035:
            score += 3.0
        elif f["avg_motion_speed"] > 0.025:
            score += 2.0
        # 手臂也在大幅摆动
        if f["avg_hand_speed"] > 0.02:
            score += 1.5
        # 重心有起伏
        if f["center_y_std"] > 0.01:
            score += 1.0
        return score

    def _score_jumping(self, f):
        """跳跃: 重心急速变化 + 双膝弯曲/伸展"""
        score = 0.0
        # 重心有明显变化
        if abs(f["center_y_velocity"]) > 0.04:
            score += 3.0
        elif abs(f["center_y_velocity"]) > 0.02:
            score += 2.0
        # 重心波动大
        if f["center_y_std"] > 0.02:
            score += 2.5
        elif f["center_y_std"] > 0.01:
            score += 1.5
        # 高整体运动
        if f["max_motion_speed"] > 0.04:
            score += 2.0
        elif f["max_motion_speed"] > 0.02:
            score += 1.0
        # 膝关节有弯曲 (蓄力/着地)
        if f["avg_knee_angle"] < 150:
            score += 1.5
        return score

    def _score_waving(self, f):
        """挥手: 手高于肩 + 水平周期性运动"""
        score = 0.0
        # 手腕高于肩
        if f["max_wrist_above_shoulder"] > 0.05:
            score += 3.0
        elif f["max_wrist_above_shoulder"] > 0.02:
            score += 1.5
        # 手腕水平摆动
        wrist_range = max(f["left_wrist_x_range"], f["right_wrist_x_range"])
        if wrist_range > 0.06:
            score += 2.5
        elif wrist_range > 0.03:
            score += 1.5
        # 周期性方向变化
        if f["wrist_oscillation"] >= 3:
            score += 3.0
        elif f["wrist_oscillation"] >= 2:
            score += 2.0
        elif f["wrist_oscillation"] >= 1:
            score += 1.0
        # 手速度较高
        if f["max_hand_speed"] > 0.02:
            score += 1.5
        # 身体整体运动低 (只有手在动)
        if f["avg_foot_speed"] < 0.01:
            score += 1.0
        return score

    def _score_bending(self, f):
        """弯腰: 上半身前倾 + 身体倾斜角大"""
        score = 0.0
        # 身体前倾
        if f["body_tilt"] > 45:
            score += 3.5
        elif f["body_tilt"] > 35:
            score += 2.5
        elif f["body_tilt"] > 25:
            score += 1.5
        # 髋关节角度减小
        if f["avg_hip_angle"] < 120:
            score += 3.0
        elif f["avg_hip_angle"] < 140:
            score += 2.0
        elif f["avg_hip_angle"] < 155:
            score += 1.0
        # 膝关节基本伸直 (弯腰不弯膝)
        if f["avg_knee_angle"] > 140:
            score += 1.5
        # 低运动速度
        if f["avg_motion_speed"] < 0.02:
            score += 1.0
        return score

    def _score_raising_hand(self, f):
        """举手: 手腕高于头部"""
        score = 0.0
        # 手腕高于头部
        if f["max_wrist_above_head"] > 0.08:
            score += 4.0
        elif f["max_wrist_above_head"] > 0.04:
            score += 2.5
        elif f["max_wrist_above_head"] > 0.01:
            score += 1.0
        # 手臂伸展 (肘角大)
        max_elbow = max(f["left_elbow_angle"], f["right_elbow_angle"])
        if max_elbow > 150:
            score += 2.0
        elif max_elbow > 120:
            score += 1.0
        # 身体直立
        if f["body_tilt"] < 15:
            score += 1.5
        elif f["body_tilt"] < 25:
            score += 0.5
        # 脚部运动低
        if f["avg_foot_speed"] < 0.01:
            score += 1.5
        # 手腕水平运动小 (区别于挥手)
        wrist_range = max(f["left_wrist_x_range"], f["right_wrist_x_range"])
        if wrist_range < 0.04:
            score += 1.0
        return score

    def _score_kicking(self, f):
        """踢腿: 单腿脚踝高度异常 + 膝角快速变化"""
        score = 0.0
        # 两脚高度差大
        if f["ankle_height_diff"] > 0.12:
            score += 3.5
        elif f["ankle_height_diff"] > 0.08:
            score += 2.5
        elif f["ankle_height_diff"] > 0.05:
            score += 1.5
        # 脚踝高于正常位置
        if f["max_ankle_above_hip"] > 0:
            score += 2.0
        # 脚部高速运动
        if f["max_foot_speed"] > 0.03:
            score += 2.5
        elif f["max_foot_speed"] > 0.02:
            score += 1.5
        # 膝角差异 (一条腿弯, 一条腿直)
        knee_diff = abs(f["left_knee_angle"] - f["right_knee_angle"])
        if knee_diff > 30:
            score += 2.0
        elif knee_diff > 15:
            score += 1.0
        # 身体相对直立
        if f["body_tilt"] < 30:
            score += 0.5
        return score

    def _score_falling(self, f):
        """跌倒: 重心急速下降 + 身体轴倾斜严重"""
        score = 0.0
        # 重心急速下降 (正值 = 下降)
        if f["center_y_velocity"] > 0.06:
            score += 4.0
        elif f["center_y_velocity"] > 0.03:
            score += 2.5
        elif f["center_y_velocity"] > 0.015:
            score += 1.0
        # 身体严重倾斜
        if f["body_tilt"] > 60:
            score += 4.0
        elif f["body_tilt"] > 45:
            score += 2.5
        elif f["body_tilt"] > 35:
            score += 1.0
        # 高整体运动速度 (跌倒过程中)
        if f["max_motion_speed"] > 0.04:
            score += 2.0
        elif f["max_motion_speed"] > 0.02:
            score += 1.0
        # 重心波动大
        if f["center_y_std"] > 0.025:
            score += 1.5
        return score

    def reset(self):
        """重置缓冲区"""
        self.landmark_buffer.clear()
        self.center_y_buffer.clear()
        self.prediction_history.clear()
