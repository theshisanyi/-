"""
基于深度学习的人体行为识别系统 (HARS) — 全局配置文件

包含动作类别定义、模型超参数、路径设置、骨架连接关系、颜色调色板等。
"""

import os

# ======================== 项目路径 ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback_data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 自动创建目录
for d in [MODEL_DIR, FEEDBACK_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ======================== 动作类别 ========================
ACTION_CLASSES = {
    0: "行走",     # Walking
    1: "跑步",     # Running
    2: "跳跃",     # Jumping
    3: "坐下",     # Sitting
    4: "站立",     # Standing
    5: "挥手",     # Waving
    6: "弯腰",     # Bending
    7: "举手",     # Raising Hand
    8: "踢腿",     # Kicking
    9: "跌倒",     # Falling
}
ACTION_CLASSES_EN = {
    0: "Walking", 1: "Running", 2: "Jumping", 3: "Sitting", 4: "Standing",
    5: "Waving", 6: "Bending", 7: "Raising Hand", 8: "Kicking", 9: "Falling",
}
NUM_CLASSES = len(ACTION_CLASSES)

# ======================== MediaPipe 配置 ========================
MEDIAPIPE_CONFIG = {
    "static_image_mode": False,
    "model_complexity": 1,          # 0=轻量, 1=标准, 2=重量级
    "smooth_landmarks": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}
NUM_KEYPOINTS = 33    # MediaPipe Pose 全部33个关键点
KEYPOINT_DIM = 3      # X, Y, Z

# ======================== 骨架连接关系 (MediaPipe 33点) ========================
SKELETON_CONNECTIONS = [
    # 面部
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # 上半身
    (11, 12),   # 双肩
    (11, 13), (13, 15),   # 左臂
    (12, 14), (14, 16),   # 右臂
    # 手部
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    # 躯干
    (11, 23), (12, 24), (23, 24),
    # 下半身
    (23, 25), (25, 27),   # 左腿
    (24, 26), (26, 28),   # 右腿
    # 足部
    (27, 29), (27, 31),
    (28, 30), (28, 32),
]

# 关键骨骼点索引 (用于规则分类器)
KEYPOINT_INDICES = {
    "nose": 0,
    "left_eye_inner": 1,  "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8,
    "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

# ======================== 模型超参数 ========================
MODEL_CONFIG = {
    "num_keypoints": NUM_KEYPOINTS,       # V = 33
    "input_channels": 6,                  # C = 6 (x,y,z + vx,vy,vz)
    "coord_channels": 3,                  # 坐标维度
    "velocity_channels": 3,               # 速度维度
    "embed_dim": 128,                     # 嵌入维度 d
    "num_stgcn_layers": 2,                # ST-GCN 层数
    "temporal_kernel_size": 5,            # 时序卷积核大小
    "num_transformer_layers": 3,          # Transformer 层数
    "num_heads": 8,                       # 注意力头数
    "ffn_dim": 512,                       # FFN 隐藏层维度
    "dropout": 0.1,                       # Dropout 率
    "num_classes": NUM_CLASSES,           # 分类数 = 10
}

# ======================== 训练配置 ========================
TRAIN_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "max_sequence_length": 64,            # 最大序列长度 T_max
    "min_sequence_length": 16,            # 最小序列长度 T_min
}

# ======================== 预训练 (SimCLR) 配置 ========================
PRETRAIN_CONFIG = {
    "batch_size": 64,
    "learning_rate": 3e-4,
    "epochs": 50,
    "temperature": 0.07,                  # NT-Xent 温度参数 τ
    "projection_dim": 64,                 # 投影头维度
}

# ======================== 推理配置 ========================
INFERENCE_CONFIG = {
    "buffer_size": 60,                    # 帧缓冲区大小
    "smoothing_window": 5,                # 预测平滑窗口
    "onnx_model_path": os.path.join(MODEL_DIR, "hars_model.onnx"),
}

# ======================== 增量学习 (EWC) 配置 ========================
EWC_CONFIG = {
    "buffer_capacity": 100,               # 反馈缓冲区容量
    "ewc_lambda": 5000,                   # EWC 正则化强度
    "finetune_epochs": 5,                 # 微调轮数
    "finetune_lr": 1e-4,                  # 微调学习率
}

# ======================== 时序自适应采样配置 ========================
SAMPLING_CONFIG = {
    "max_frames": 64,                     # 最大帧数
    "min_frames": 16,                     # 最小帧数
    "alpha": 0.5,                         # 运动量权重因子
}

# ======================== 颜色调色板 ========================
# BGR 格式 (OpenCV)
COLORS = {
    "primary":        (235, 160, 72),     # 蓝紫色
    "secondary":      (72, 235, 160),     # 绿色
    "accent":         (72, 160, 235),     # 橙黄色
    "skeleton":       (0, 255, 128),      # 荧光绿
    "keypoint":       (0, 200, 255),      # 橙色
    "text_bg":        (40, 40, 40),       # 深灰
    "text":           (255, 255, 255),    # 白色
    "confidence_high":(0, 200, 0),        # 高置信度：绿
    "confidence_mid": (0, 200, 200),      # 中置信度：黄
    "confidence_low": (0, 0, 200),        # 低置信度：红
}

# 动作对应颜色 (BGR)
ACTION_COLORS = [
    (255, 180, 0),    # 行走 - 蓝
    (0, 140, 255),    # 跑步 - 橙
    (0, 255, 128),    # 跳跃 - 绿
    (255, 100, 100),  # 坐下 - 浅蓝
    (200, 200, 200),  # 站立 - 灰
    (255, 0, 255),    # 挥手 - 粉
    (0, 255, 255),    # 弯腰 - 黄
    (128, 0, 255),    # 举手 - 紫
    (100, 255, 100),  # 踢腿 - 浅绿
    (0, 0, 255),      # 跌倒 - 红 (警告色)
]

# ======================== GUI 样式表 ========================
GUI_STYLESHEET = """
QMainWindow {
    background-color: #0f0f1a;
}
QTabWidget::pane {
    border: 1px solid #2a2a4a;
    background: #0f0f1a;
    border-radius: 8px;
}
QTabBar::tab {
    background: #1a1a2e;
    color: #8888aa;
    padding: 12px 28px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-size: 14px;
    font-weight: bold;
}
QTabBar::tab:selected {
    background: #16213e;
    color: #00d4ff;
    border-bottom: 3px solid #00d4ff;
}
QTabBar::tab:hover {
    background: #1f1f3a;
    color: #aaaacc;
}
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a73e8, stop:1 #6c5ce7);
    color: white;
    border: none;
    padding: 10px 24px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2b84f9, stop:1 #7d6df8);
}
QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f5cbf, stop:1 #5b4bc6);
}
QPushButton:disabled {
    background: #333355;
    color: #666688;
}
QPushButton#btnDanger {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e74c3c, stop:1 #c0392b);
}
QPushButton#btnDanger:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ff6b6b, stop:1 #d63031);
}
QPushButton#btnSuccess {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00b894, stop:1 #00a381);
}
QPushButton#btnSuccess:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00d4aa, stop:1 #00c49a);
}
QLabel {
    color: #ccccdd;
    font-size: 13px;
}
QLabel#titleLabel {
    color: #00d4ff;
    font-size: 28px;
    font-weight: bold;
}
QLabel#subtitleLabel {
    color: #8888aa;
    font-size: 14px;
}
QLabel#actionLabel {
    color: #00ff88;
    font-size: 36px;
    font-weight: bold;
}
QLabel#fpsLabel {
    color: #ffaa00;
    font-size: 16px;
    font-weight: bold;
}
QLabel#statusLabel {
    color: #aaaacc;
    font-size: 12px;
}
QComboBox {
    background: #1a1a2e;
    color: #ccccdd;
    border: 1px solid #2a2a4a;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
}
QComboBox:hover {
    border: 1px solid #00d4ff;
}
QComboBox QAbstractItemView {
    background: #1a1a2e;
    color: #ccccdd;
    selection-background-color: #2a4a6e;
}
QProgressBar {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 4px;
    text-align: center;
    color: white;
    font-size: 11px;
    height: 22px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a73e8, stop:1 #00d4ff);
    border-radius: 3px;
}
QGroupBox {
    color: #ccccdd;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 20px;
    font-weight: bold;
    font-size: 13px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    color: #00d4ff;
}
QTextBrowser {
    background: #12122a;
    color: #ccccdd;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 8px;
    font-size: 13px;
}
QStatusBar {
    background: #0a0a1a;
    color: #8888aa;
    border-top: 1px solid #2a2a4a;
    font-size: 12px;
}
QScrollBar:vertical {
    background: #0f0f1a;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #2a2a4a;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #3a3a5a;
}
"""
