"""
基于深度学习的人体行为识别系统 (HARS) — PyQt5 主程序

功能模块:
  Tab 1 - 实时识别: 摄像头视频流 + 骨骼绘制 + 动作标签 + 置信度
  Tab 2 - 视频分析: 视频文件上传 + 逐帧分析 + 行为统计
  Tab 3 - 反馈与增量学习: 用户纠正 + EWC在线微调 + 模型热更新
  Tab 4 - 模型架构展示: 网络结构图 + 技术原理说明

运行方式:
  python main_system.py
"""

import sys
import os
import time
import numpy as np
from collections import deque

# ★ Windows DLL冲突修复: onnxruntime 必须在 torch 之前导入
# 否则 torch 的 DLL 搜索路径会导致 onnxruntime 加载失败 (WinError 1114)
try:
    import onnxruntime
except ImportError:
    pass

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QProgressBar,
    QGroupBox, QFileDialog, QStatusBar, QSplitter,
    QTextBrowser, QSlider, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon

import cv2

# 本地模块
from config import (
    ACTION_CLASSES, ACTION_CLASSES_EN, NUM_CLASSES,
    ACTION_COLORS, COLORS, GUI_STYLESHEET, INFERENCE_CONFIG, MODEL_DIR
)
from utils import (
    draw_skeleton, draw_action_label, draw_fps,
    smooth_predictions, FPSCounter
)
from data_preprocessor import DataPreprocessor

try:
    from data_preprocessor import MediaPipeEstimator
    MEDIAPIPE_OK = True
except Exception:
    MEDIAPIPE_OK = False

from inference_engine import HybridEngine
from incremental_learner import FeedbackBuffer, BackgroundUpdater

# 尝试加载 matplotlib (用于图表)
try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


# ================================================================
# 视频采集线程
# ================================================================

class VideoThread(QThread):
    """
    视频采集与处理线程

    负责从摄像头或视频文件读取帧, 进行姿态估计和行为识别,
    通过信号将处理结果发送给主线程更新GUI
    """
    frame_ready = pyqtSignal(np.ndarray)           # 处理后的帧
    result_ready = pyqtSignal(int, float, object)   # action_id, confidence, probs
    landmarks_ready = pyqtSignal(object)            # 原始关键点
    fps_updated = pyqtSignal(float)                 # FPS
    error_occurred = pyqtSignal(str)

    def __init__(self, source=0):
        super().__init__()
        self.source = source
        self.running = False
        self.paused = False
        self.mode = "rule"  # "rule" / "dl" / "hybrid"

        self.estimator = None
        self.engine = None
        self.fps_counter = FPSCounter()
        self.prediction_history = deque(maxlen=10)

    def setup(self):
        """初始化姿态估计和推理引擎"""
        if MEDIAPIPE_OK:
            self.estimator = MediaPipeEstimator()
        self.engine = HybridEngine()

    def run(self):
        """线程主循环"""
        try:
            self.setup()
        except Exception as e:
            error_msg = f"初始化失败: {e}"
            print(error_msg)
            self.error_occurred.emit(str(e))
            return
        self.running = True
        cap = None
        try:

            # (修复 Windows 平台摄像头在子线程中崩溃的问题: 使用 CAP_DSHOW 后端)
            if isinstance(self.source, int):
                if os.name == 'nt':
                    cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(self.source)

            if not cap.isOpened():
                print(f"[错误] 无法打开视频源: {self.source}")
                # 回退到默认后端
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    return

            # 设置摄像头参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while self.running:
                if self.paused:
                    self.msleep(50)
                    continue

                ret, frame = cap.read()
                if not ret:
                    if isinstance(self.source, str):
                        # 视频文件播放完毕, 重新开始
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        self.msleep(10)
                        continue

                self.fps_counter.tick()

                # 姿态估计
                landmarks = None
                display_frame = frame.copy()
                if self.estimator is not None:
                    landmarks = self.estimator.estimate(display_frame)

                if landmarks is not None:
                    display_frame = draw_skeleton(display_frame, landmarks)
                    self.landmarks_ready.emit(landmarks)

                    # 行为识别
                    action_id, confidence, probs, _mode = self.engine.predict(
                        landmarks, mode=self.mode
                    )

                    self.prediction_history.append((action_id, confidence))
                    smoothed_id, smoothed_conf = smooth_predictions(
                        list(self.prediction_history)
                    )

                    # 绘制结果
                    display_frame = draw_action_label(
                        display_frame, smoothed_id, smoothed_conf
                    )

                    self.result_ready.emit(smoothed_id, smoothed_conf, probs)

                # 绘制FPS
                fps = self.fps_counter.get_fps()
                display_frame = draw_fps(display_frame, fps)
                self.fps_updated.emit(fps)

                self.frame_ready.emit(display_frame)
                self.msleep(1)  # 避免过快

        except Exception as e:
            print(f"[错误] 视频线程异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cap is not None:
                cap.release()
            if self.estimator:
                self.estimator.release()

    def stop(self):
        self.running = False
        self.wait(3000)

    def set_mode(self, mode):
        self.mode = mode




# ================================================================
# 实时识别 Tab
# ================================================================

class RealTimeTab(QWidget):
    """Tab 1: 实时行为识别"""

    feedback_signal = pyqtSignal(object, int)  # landmarks, correct_label

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.video_thread = None
        self.current_landmarks = None
        self.current_action = 4
        self.current_confidence = 0.0
        self.current_probs = None
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # ---- 左侧: 视频显示 ----
        left_panel = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background: #0a0a1a;
                border: 2px solid #2a2a4a;
                border-radius: 12px;
            }
        """)
        self.video_label.setText("📷 点击「开始识别」启动摄像头")
        left_panel.addWidget(self.video_label)

        # 控制按钮
        ctrl_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ 开始识别")
        self.btn_start.setMinimumHeight(42)
        self.btn_start.clicked.connect(self.start_recognition)

        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setObjectName("btnDanger")
        self.btn_stop.setMinimumHeight(42)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_recognition)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["规则模式", "深度学习模式", "混合模式"])
        self.mode_combo.setMinimumHeight(42)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(QLabel("识别模式:"))
        ctrl_layout.addWidget(self.mode_combo)
        left_panel.addLayout(ctrl_layout)

        layout.addLayout(left_panel, stretch=3)

        # ---- 右侧: 结果面板 ----
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)

        # 当前动作
        action_group = QGroupBox("◆ 当前识别结果")
        action_layout = QVBoxLayout(action_group)
        self.action_label = QLabel("等待识别...")
        self.action_label.setObjectName("actionLabel")
        self.action_label.setAlignment(Qt.AlignCenter)
        action_layout.addWidget(self.action_label)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("置信度: %p%")
        action_layout.addWidget(self.confidence_bar)

        right_panel.addWidget(action_group)

        # FPS
        fps_group = QGroupBox("◆ 系统状态")
        fps_layout = QVBoxLayout(fps_group)
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("fpsLabel")
        fps_layout.addWidget(self.fps_label)

        mode_label_text = QLabel("当前模式: 规则分类器")
        mode_label_text.setObjectName("statusLabel")
        self.mode_status_label = mode_label_text
        fps_layout.addWidget(mode_label_text)
        right_panel.addWidget(fps_group)

        # 概率分布
        prob_group = QGroupBox("◆ 各行为概率分布")
        prob_layout = QVBoxLayout(prob_group)
        self.prob_bars = {}
        for i in range(NUM_CLASSES):
            h = QHBoxLayout()
            name_label = QLabel(f"{ACTION_CLASSES[i]}")
            name_label.setFixedWidth(50)
            name_label.setStyleSheet("font-size: 12px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(18)
            bar.setFormat("%p%")
            bar.setStyleSheet(f"""
                QProgressBar::chunk {{
                    background: rgb({ACTION_COLORS[i][2]},{ACTION_COLORS[i][1]},{ACTION_COLORS[i][0]});
                    border-radius: 2px;
                }}
            """)
            h.addWidget(name_label)
            h.addWidget(bar)
            prob_layout.addLayout(h)
            self.prob_bars[i] = bar
        right_panel.addWidget(prob_group)

        # 反馈按钮
        fb_group = QGroupBox("◆ 结果反馈")
        fb_layout = QVBoxLayout(fb_group)
        fb_btn_layout = QHBoxLayout()

        self.btn_correct = QPushButton("✓ 正确")
        self.btn_correct.setObjectName("btnSuccess")
        self.btn_correct.setMinimumHeight(36)
        self.btn_correct.clicked.connect(self.on_correct)

        self.btn_wrong = QPushButton("✗ 错误")
        self.btn_wrong.setObjectName("btnDanger")
        self.btn_wrong.setMinimumHeight(36)
        self.btn_wrong.clicked.connect(self.on_wrong)

        fb_btn_layout.addWidget(self.btn_correct)
        fb_btn_layout.addWidget(self.btn_wrong)
        fb_layout.addLayout(fb_btn_layout)

        corr_layout = QHBoxLayout()
        corr_layout.addWidget(QLabel("纠正为:"))
        self.correct_combo = QComboBox()
        for i in range(NUM_CLASSES):
            self.correct_combo.addItem(f"{ACTION_CLASSES[i]} ({ACTION_CLASSES_EN[i]})")
        corr_layout.addWidget(self.correct_combo)
        fb_layout.addLayout(corr_layout)
        right_panel.addWidget(fb_group)

        right_panel.addStretch()
        layout.addLayout(right_panel, stretch=2)

    def start_recognition(self):
        if self.video_thread is not None and self.video_thread.isRunning():
            return

        self.video_thread = VideoThread(source=0)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.result_ready.connect(self.update_result)
        self.video_thread.landmarks_ready.connect(self.update_landmarks)
        self.video_thread.fps_updated.connect(self.update_fps)
        self.video_thread.error_occurred.connect(self.on_video_error)

        modes = ["rule", "dl", "hybrid"]
        self.video_thread.set_mode(modes[self.mode_combo.currentIndex()])

        self.video_thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_recognition(self):
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.video_label.setText("📷 已停止，点击「开始识别」重新启动")

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def update_frame(self, frame):
        self._show_frame(frame)

    def update_result(self, action_id, confidence, probs):
        self.current_action = action_id
        self.current_confidence = confidence
        self.current_probs = probs

        name = ACTION_CLASSES.get(action_id, "未知")
        en_name = ACTION_CLASSES_EN.get(action_id, "Unknown")
        self.action_label.setText(f"{name}\n{en_name}")

        color = ACTION_COLORS[action_id] if action_id < len(ACTION_COLORS) else (200, 200, 200)
        self.action_label.setStyleSheet(
            f"color: rgb({color[2]},{color[1]},{color[0]}); "
            f"font-size: 36px; font-weight: bold;"
        )

        self.confidence_bar.setValue(int(confidence * 100))

        if isinstance(probs, np.ndarray) and len(probs) == NUM_CLASSES:
            for i in range(NUM_CLASSES):
                self.prob_bars[i].setValue(int(probs[i] * 100))

    def update_landmarks(self, landmarks):
        self.current_landmarks = landmarks
        if self.main_window and landmarks is not None:
            self.main_window.feedback_frame_buffer.append(landmarks)

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def on_video_error(self, error_msg):
        self.stop_recognition()
        self.video_label.setText(f"⚠ 初始化失败: {error_msg}")

    def on_mode_changed(self, index):
        modes = ["rule", "dl", "hybrid"]
        mode_names = ["规则分类器", "深度学习模型", "混合模式"]
        if self.video_thread is not None:
            self.video_thread.set_mode(modes[index])
        self.mode_status_label.setText(f"当前模式: {mode_names[index]}")

    def on_correct(self):
        if self.main_window and hasattr(self.main_window, 'on_feedback'):
            self.main_window.on_feedback(self.current_landmarks, self.current_action, True)

    def on_wrong(self):
        correct_label = self.correct_combo.currentIndex()
        if self.main_window and hasattr(self.main_window, 'on_feedback'):
            self.main_window.on_feedback(self.current_landmarks, correct_label, False)


# ================================================================
# 视频分析 Tab
# ================================================================

class VideoAnalysisTab(QWidget):
    """Tab 2: 视频文件分析"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.video_path = None
        self.analysis_thread = None

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # 上部: 视频选择
        top_layout = QHBoxLayout()
        self.btn_select = QPushButton("📁 选择视频文件")
        self.btn_select.setMinimumHeight(42)
        self.btn_select.clicked.connect(self.select_video)

        self.path_label = QLabel("未选择文件")
        self.path_label.setStyleSheet("color: #8888aa;")

        self.btn_analyze = QPushButton("🔍 开始分析")
        self.btn_analyze.setMinimumHeight(42)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.start_analysis)

        top_layout.addWidget(self.btn_select)
        top_layout.addWidget(self.path_label, stretch=1)
        top_layout.addWidget(self.btn_analyze)

        self.analysis_mode_combo = QComboBox()
        self.analysis_mode_combo.addItems(["规则模式", "深度学习模式", "混合模式"])
        self.analysis_mode_combo.setMinimumHeight(36)
        top_layout.addWidget(QLabel("推理模式:"))
        top_layout.addWidget(self.analysis_mode_combo)

        layout.addLayout(top_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 中部: 视频 + 结果
        mid_layout = QHBoxLayout()

        # 视频预览
        self.preview_label = QLabel("视频预览区域")
        self.preview_label.setMinimumSize(480, 360)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                background: #0a0a1a;
                border: 2px solid #2a2a4a;
                border-radius: 12px;
                color: #555;
            }
        """)
        mid_layout.addWidget(self.preview_label, stretch=2)

        # 分析结果
        result_layout = QVBoxLayout()

        self.result_text = QTextBrowser()
        self.result_text.setMinimumWidth(300)
        self.result_text.setHtml("""
            <div style='color:#8888aa; padding:20px; text-align:center;'>
            <h3>📊 分析结果</h3>
            <p>选择视频文件后点击"开始分析"</p>
            </div>
        """)
        result_layout.addWidget(self.result_text)

        # 图表
        if MATPLOTLIB_OK:
            self.chart_figure = Figure(figsize=(4, 3), facecolor='#0f0f1a')
            self.chart_canvas = FigureCanvas(self.chart_figure)
            self.chart_canvas.setMinimumHeight(250)
            result_layout.addWidget(self.chart_canvas)

        mid_layout.addLayout(result_layout, stretch=1)
        layout.addLayout(mid_layout)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if path:
            self.video_path = path
            self.path_label.setText(os.path.basename(path))
            self.btn_analyze.setEnabled(True)

            # 预览第一帧
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            if ret:
                self._show_frame(frame)
            cap.release()

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        scaled = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

    def start_analysis(self):
        if not self.video_path:
            return

        self.btn_analyze.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        modes = ["rule", "dl", "hybrid"]
        mode = modes[self.analysis_mode_combo.currentIndex()]
        self.analysis_thread = VideoAnalysisThread(self.video_path, mode=mode)
        self.analysis_thread.progress.connect(self.on_progress)
        self.analysis_thread.frame_processed.connect(self._show_frame)
        self.analysis_thread.finished_signal.connect(self.on_analysis_done)
        self.analysis_thread.start()

    def on_progress(self, current, total):
        pct = int(current / max(total, 1) * 100)
        self.progress_bar.setValue(pct)

    def on_analysis_done(self, results):
        self.progress_bar.setVisible(False)
        self.btn_analyze.setEnabled(True)

        if not results:
            self.result_text.setHtml("<p style='color:red;'>分析失败</p>")
            return

        # 统计
        action_counts = {}
        for action_id in results:
            name = ACTION_CLASSES.get(action_id, "未知")
            action_counts[name] = action_counts.get(name, 0) + 1

        total = sum(action_counts.values())

        html = "<div style='color:#ccccdd; padding:10px;'>"
        html += "<h3 style='color:#00d4ff;'>📊 视频分析报告</h3>"
        html += f"<p>总帧数: {total}</p><hr>"

        for name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            pct = count / max(total, 1) * 100
            html += f"<p><b>{name}</b>: {count} 帧 ({pct:.1f}%)</p>"

        html += "</div>"
        self.result_text.setHtml(html)

        # 绘制饼图
        if MATPLOTLIB_OK and action_counts:
            self.chart_figure.clear()
            ax = self.chart_figure.add_subplot(111)
            labels = list(action_counts.keys())
            sizes = list(action_counts.values())
            # 使用动作颜色
            colors_plt = []
            for name in labels:
                idx = list(ACTION_CLASSES.values()).index(name) if name in ACTION_CLASSES.values() else 0
                c = ACTION_COLORS[idx]
                colors_plt.append((c[2]/255, c[1]/255, c[0]/255))

            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct='%1.1f%%',
                colors=colors_plt, textprops={'color': '#ccccdd', 'fontsize': 9}
            )
            for t in autotexts:
                t.set_color('white')
                t.set_fontsize(8)
            ax.set_title('行为分布统计', color='#00d4ff', fontsize=13, fontweight='bold')
            self.chart_figure.patch.set_facecolor('#0f0f1a')
            self.chart_canvas.draw()


class VideoAnalysisThread(QThread):
    """视频分析后台线程"""
    progress = pyqtSignal(int, int)
    frame_processed = pyqtSignal(np.ndarray)
    finished_signal = pyqtSignal(list)

    def __init__(self, path, mode="rule"):
        super().__init__()
        self.path = path
        self.mode = mode
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        results = []
        estimator = None
        cap = None
        try:
            if not MEDIAPIPE_OK:
                self.finished_signal.emit([])
                return

            estimator = MediaPipeEstimator()
            engine = HybridEngine()
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                print(f"[视频分析] 无法打开视频: {self.path}")
                self.finished_signal.emit([])
                return

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if self._stop_flag:
                    break

                landmarks = estimator.estimate(frame)
                if landmarks is not None:
                    action_id, conf, probs, _m = engine.predict(landmarks, mode=self.mode)
                    results.append(action_id)

                    display = frame.copy()
                    draw_skeleton(display, landmarks)
                    draw_action_label(display, action_id, conf)

                    if idx % 5 == 0:
                        self.frame_processed.emit(display)

                idx += 1
                if idx % 3 == 0:
                    self.progress.emit(idx, total)
        except Exception as e:
            print(f"[视频分析] 错误: {e}")
        finally:
            if cap is not None:
                cap.release()
            if estimator is not None:
                estimator.release()

        self.finished_signal.emit(results)


# ================================================================
# 反馈与增量学习 Tab
# ================================================================

class FeedbackTab(QWidget):
    """Tab 3: 反馈与增量学习"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # 反馈状态
        status_group = QGroupBox("◆ 增量学习状态")
        status_layout = QGridLayout(status_group)

        status_layout.addWidget(QLabel("反馈样本数:"), 0, 0)
        self.feedback_count_label = QLabel("0 / 100")
        self.feedback_count_label.setStyleSheet("color: #00d4ff; font-size: 18px; font-weight: bold;")
        status_layout.addWidget(self.feedback_count_label, 0, 1)

        status_layout.addWidget(QLabel("模型版本:"), 1, 0)
        self.model_version_label = QLabel("v1.0 (初始)")
        self.model_version_label.setStyleSheet("color: #00ff88; font-size: 14px;")
        status_layout.addWidget(self.model_version_label, 1, 1)

        status_layout.addWidget(QLabel("学习状态:"), 2, 0)
        self.learn_status_label = QLabel("⏸ 等待反馈数据")
        self.learn_status_label.setStyleSheet("color: #ffaa00; font-size: 14px;")
        status_layout.addWidget(self.learn_status_label, 2, 1)

        layout.addWidget(status_group)

        # 反馈缓冲区进度
        progress_group = QGroupBox("◆ 反馈缓冲区")
        progress_layout = QVBoxLayout(progress_group)
        self.feedback_progress = QProgressBar()
        self.feedback_progress.setRange(0, 100)
        self.feedback_progress.setValue(0)
        self.feedback_progress.setFormat("缓冲区: %v / 100 (%p%)")
        progress_layout.addWidget(self.feedback_progress)

        self.btn_force_train = QPushButton("🧠 立即触发微调")
        self.btn_force_train.setMinimumHeight(42)
        self.btn_force_train.clicked.connect(self.force_train)
        progress_layout.addWidget(self.btn_force_train)

        layout.addWidget(progress_group)

        # 微调进度
        train_group = QGroupBox("◆ 微调训练进度")
        train_layout = QVBoxLayout(train_group)
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        self.train_progress.setFormat("训练: %p%")
        train_layout.addWidget(self.train_progress)

        self.train_log = QTextBrowser()
        self.train_log.setMaximumHeight(200)
        train_layout.addWidget(self.train_log)

        layout.addWidget(train_group)

        # EWC 说明
        info_group = QGroupBox("◆ EWC 增量学习原理")
        info_layout = QVBoxLayout(info_group)
        info_text = QTextBrowser()
        info_text.setHtml("""
        <div style='color:#ccccdd; padding:8px;'>
        <h4 style='color:#00d4ff;'>弹性权重巩固 (Elastic Weight Consolidation)</h4>
        <p>EWC通过Fisher信息矩阵识别对旧任务重要的参数, 在学习新任务时
        对这些参数施加正则化约束, 从而在学习新知识的同时避免灾难性遗忘。</p>
        <p><b>损失函数:</b></p>
        <p style='color:#00ff88; font-family:monospace;'>
        L_total = L_task + λ × Σ F_i × (θ_i - θ*_i)²
        </p>
        <ul>
        <li><b>L_task</b>: 新任务交叉熵损失</li>
        <li><b>F_i</b>: Fisher信息矩阵 (参数重要性)</li>
        <li><b>θ*_i</b>: 旧任务最优参数</li>
        <li><b>λ</b>: 正则化强度 (默认: 5000)</li>
        </ul>
        <p><b>工作流程:</b> 用户纠正结果 → 缓存反馈 → 达到阈值 → 后台EWC微调 → 模型热更新</p>
        </div>
        """)
        info_text.setMaximumHeight(280)
        info_layout.addWidget(info_text)
        layout.addWidget(info_group)

        layout.addStretch()

    def update_feedback_count(self, count, capacity):
        self.feedback_count_label.setText(f"{count} / {capacity}")
        pct = int(count / max(capacity, 1) * 100)
        self.feedback_progress.setValue(pct)

    def on_train_progress(self, epoch, total, loss):
        pct = int(epoch / max(total, 1) * 100)
        self.train_progress.setValue(pct)
        self.train_log.append(f"Epoch {epoch}/{total} | Loss: {loss:.4f}")
        self.learn_status_label.setText(f"🔄 训练中... Epoch {epoch}/{total}")

    def on_train_complete(self, success):
        if success:
            self.learn_status_label.setText("✓ 微调完成, 模型已更新")
            self.learn_status_label.setStyleSheet("color: #00ff88; font-size: 14px;")
            self.train_log.append("✓ 模型微调成功, 已热更新!")
        else:
            self.learn_status_label.setText("✗ 微调失败")
            self.learn_status_label.setStyleSheet("color: #ff4444; font-size: 14px;")

    def force_train(self):
        if self.main_window and hasattr(self.main_window, 'trigger_incremental_learning'):
            self.main_window.trigger_incremental_learning(force=True)


# ================================================================
# 模型架构展示 Tab
# ================================================================

class ArchitectureTab(QWidget):
    """Tab 4: 模型架构展示"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        text = QTextBrowser()
        text.setOpenExternalLinks(True)
        text.setHtml(self._get_architecture_html())
        layout.addWidget(text)

    def _get_architecture_html(self):
        return """
        <div style='color:#ccccdd; padding:16px; font-family:Segoe UI,Microsoft YaHei,sans-serif;'>

        <h2 style='color:#00d4ff; text-align:center;'>🧠 ST-GCN + Transformer 混合模型架构</h2>
        <hr style='border-color:#2a2a4a;'>

        <h3 style='color:#00ff88;'>📐 整体架构流程</h3>
        <pre style='color:#aaddff; background:#12122a; padding:16px;
                    border-radius:8px; font-size:13px; line-height:1.6;'>
┌──────────────────────────────────────────────────┐
│  输入: (Batch, T, V=33, C=6)                     │
│  T=时间帧数, V=关节数, C=坐标(3)+速度(3)          │
└──────────────┬───────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────┐
│  多流嵌入层 (MultiStreamEmbedding)                │
│  坐标流 → Linear(3,128) → LN → ReLU              │
│  速度流 → Linear(3,128) → LN → ReLU              │
│  拼接(256) → Linear(256,128) → LN → ReLU         │
│  输出: (B, T, V, 128)                            │
└──────────────┬───────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────┐
│  ST-GCN 块 ×2                                     │
│  ┌────────────────────────────────────────────┐  │
│  │ 自适应图卷积 (AdaptiveGraphConv)            │  │
│  │ A_adaptive = A_fixed + A_learn + A_data    │  │
│  │ A_data = softmax(Q·K^T / √d)              │  │
│  │ Output = A_adaptive · X · W + BN           │  │
│  └────────────────┬───────────────────────────┘  │
│                   ▼                               │
│  ┌────────────────────────────────────────────┐  │
│  │ 时序卷积 (TemporalConv)                     │  │
│  │ Conv1D(kernel=5) + BN + ReLU + Dropout     │  │
│  └────────────────┬───────────────────────────┘  │
│                   ▼                               │
│  残差连接 + ReLU + Dropout                        │
│  输出: (B, T, V, 128)                            │
└──────────────┬───────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────┐
│  空间池化 (Spatial Pooling)                       │
│  AdaptiveAvgPool 沿关节维度V聚合                   │
│  输出: (B, T, 128)                               │
└──────────────┬───────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────┐
│  位置编码 (Positional Encoding)                    │
│  正弦余弦编码: PE(pos, 2i) = sin(pos/10000^(2i/d))│
└──────────────┬───────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────┐
│  Transformer 编码器 ×3                             │
│  ┌────────────────────────────────────────────┐  │
│  │ 多头自注意力 (8头, d=128)                    │  │
│  │ Attention(Q,K,V) = softmax(QK^T/√d)·V     │  │
│  └────────────────┬───────────────────────────┘  │
│                   ▼                               │
│  ┌────────────────────────────────────────────┐  │
│  │ 前馈网络 FFN(128→512→128)                   │  │
│  │ + LayerNorm + Dropout + Residual           │  │
│  └────────────────────────────────────────────┘  │
│  输出: (B, T, 128)                               │
└──────────────┬───────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────┐
│  全局平均池化 (Global Average Pooling)             │
│  沿时间维度T取均值                                 │
│  输出: (B, 128)                                   │
└──────────────┬───────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────┐
│  分类头 (Classification Head)                     │
│  Linear(128→64) → ReLU → Dropout                 │
│  Linear(64→10) → Softmax                         │
│  输出: 10类概率分布                                │
└──────────────────────────────────────────────────┘
        </pre>

        <hr style='border-color:#2a2a4a;'>

        <h3 style='color:#00ff88;'>🦴 MediaPipe 33个人体关键点</h3>
        <pre style='color:#aaddff; background:#12122a; padding:16px;
                    border-radius:8px; font-size:12px; line-height:1.5;'>
                    ● 0 鼻子
                   / \\
           1,2,3 ●   ● 4,5,6 (眼睛)
                7 ●   ● 8    (耳朵)
                 9 ● ● 10    (嘴角)

              11 ●─────● 12   (肩膀)
                 │     │
              13 ●     ● 14   (肘部)
                 │     │
              15 ●     ● 16   (手腕)
                /│\\   /│\\
           17,19,21  18,20,22 (手指)

              23 ●─────● 24   (髋部)
                 │     │
              25 ●     ● 26   (膝盖)
                 │     │
              27 ●     ● 28   (脚踝)
                /│    /│
            29,31●  30,32●    (脚部)
        </pre>

        <hr style='border-color:#2a2a4a;'>

        <h3 style='color:#00ff88;'>🔄 系统三层架构</h3>
        <table style='color:#ccccdd; width:100%; border-collapse:collapse;'>
        <tr style='background:#1a1a2e;'>
            <th style='padding:10px; border:1px solid #2a2a4a; color:#00d4ff;'>层次</th>
            <th style='padding:10px; border:1px solid #2a2a4a; color:#00d4ff;'>模块</th>
            <th style='padding:10px; border:1px solid #2a2a4a; color:#00d4ff;'>技术</th>
        </tr>
        <tr><td style='padding:8px; border:1px solid #2a2a4a;'>离线训练</td>
            <td style='padding:8px; border:1px solid #2a2a4a;'>SimCLR预训练 + 全监督训练</td>
            <td style='padding:8px; border:1px solid #2a2a4a;'>对比学习 + 交叉熵</td></tr>
        <tr style='background:#12122a;'>
            <td style='padding:8px; border:1px solid #2a2a4a;'>在线推理</td>
            <td style='padding:8px; border:1px solid #2a2a4a;'>MediaPipe + ONNX Runtime</td>
            <td style='padding:8px; border:1px solid #2a2a4a;'>姿态估计 + 模型推理</td></tr>
        <tr><td style='padding:8px; border:1px solid #2a2a4a;'>增量学习</td>
            <td style='padding:8px; border:1px solid #2a2a4a;'>EWC在线微调</td>
            <td style='padding:8px; border:1px solid #2a2a4a;'>Fisher正则化 + 热更新</td></tr>
        </table>

        <hr style='border-color:#2a2a4a;'>

        <h3 style='color:#00ff88;'>🎯 10种可识别行为</h3>
        <table style='color:#ccccdd; width:100%; border-collapse:collapse;'>
        <tr style='background:#1a1a2e;'>
            <th style='padding:8px; border:1px solid #2a2a4a; color:#00d4ff;'>编号</th>
            <th style='padding:8px; border:1px solid #2a2a4a; color:#00d4ff;'>中文</th>
            <th style='padding:8px; border:1px solid #2a2a4a; color:#00d4ff;'>English</th>
            <th style='padding:8px; border:1px solid #2a2a4a; color:#00d4ff;'>核心特征</th>
        </tr>
        <tr><td style='padding:6px; border:1px solid #2a2a4a;'>0</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>行走</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>Walking</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>脚步交替 + 中等速度</td></tr>
        <tr style='background:#12122a;'>
            <td style='padding:6px; border:1px solid #2a2a4a;'>1</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>跑步</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>Running</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>高频步伐 + 大幅摆臂</td></tr>
        <tr><td style='padding:6px; border:1px solid #2a2a4a;'>2</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>跳跃</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>Jumping</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>重心突变 + 膝关节弯曲</td></tr>
        <tr style='background:#12122a;'>
            <td style='padding:6px; border:1px solid #2a2a4a;'>3</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>坐下</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>Sitting</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>髋膝角减小 + 低运动</td></tr>
        <tr><td style='padding:6px; border:1px solid #2a2a4a;'>4</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>站立</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>Standing</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>直立 + 静止</td></tr>
        <tr style='background:#12122a;'>
            <td style='padding:6px; border:1px solid #2a2a4a;'>5</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>挥手</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>Waving</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>手高于肩 + 水平振荡</td></tr>
        <tr><td style='padding:6px; border:1px solid #2a2a4a;'>6</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>弯腰</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>Bending</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>躯干前倾 + 髋角减小</td></tr>
        <tr style='background:#12122a;'>
            <td style='padding:6px; border:1px solid #2a2a4a;'>7</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>举手</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>Raising Hand</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>手腕高于头部</td></tr>
        <tr><td style='padding:6px; border:1px solid #2a2a4a;'>8</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>踢腿</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>Kicking</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>单腿高速运动 + 膝角突变</td></tr>
        <tr style='background:#12122a;'>
            <td style='padding:6px; border:1px solid #2a2a4a;'>9</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>跌倒</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>Falling</td>
            <td style='padding:6px; border:1px solid #2a2a4a;'>重心急降 + 躯干严重倾斜</td></tr>
        </table>

        <hr style='border-color:#2a2a4a;'>

        <h3 style='color:#00ff88;'>⚡ 创新技术要点</h3>
        <ul style='line-height:2;'>
        <li><b style='color:#ffaa00;'>自适应图卷积:</b> 结合固定拓扑 + 可学习 + 数据驱动注意力</li>
        <li><b style='color:#ffaa00;'>时序自适应采样:</b> 根据动作快慢调整处理帧数</li>
        <li><b style='color:#ffaa00;'>自监督预训练:</b> SimCLR对比学习提升小样本泛化</li>
        <li><b style='color:#ffaa00;'>模型压缩:</b> 结构化剪枝(30%) + INT8量化感知训练</li>
        <li><b style='color:#ffaa00;'>EWC增量学习:</b> Fisher信息矩阵约束防止灾难性遗忘</li>
        </ul>

        </div>
        """


# ================================================================
# 主窗口
# ================================================================

class MainWindow(QMainWindow):
    """HARS 系统主窗口"""

    train_progress_signal = pyqtSignal(int, int, float)
    train_complete_signal = pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于深度学习的人体行为识别系统 (HARS)")
        self.setMinimumSize(1280, 800)
        self.resize(1400, 900)

        # 反馈缓冲区和预处理器
        self.feedback_buffer = FeedbackBuffer()
        self.feedback_frame_buffer = deque(maxlen=32)
        self.preprocessor = DataPreprocessor()
        self.bg_updater = None
        self.model_version = 1

        self._init_ui()
        self._init_statusbar()

        # 状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(2000)

        self.train_progress_signal.connect(self._on_train_progress_safe)
        self.train_complete_signal.connect(self._on_train_complete_safe)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 标题栏
        header = QWidget()
        header.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0a0a1a, stop:0.5 #12122e, stop:1 #0a0a1a);
                border-bottom: 1px solid #2a2a4a;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 12, 24, 12)

        title = QLabel("🧠 基于深度学习的人体行为识别系统")
        title.setObjectName("titleLabel")
        header_layout.addWidget(title)

        header_layout.addStretch()

        subtitle = QLabel("HARS — Human Action Recognition System")
        subtitle.setObjectName("subtitleLabel")
        header_layout.addWidget(subtitle)

        main_layout.addWidget(header)

        # Tab 组件
        self.tabs = QTabWidget()
        self.tab_realtime = RealTimeTab(self)
        self.tab_video = VideoAnalysisTab(self)
        self.tab_feedback = FeedbackTab(self)
        self.tab_arch = ArchitectureTab(self)

        self.tabs.addTab(self.tab_realtime, "📷 实时识别")
        self.tabs.addTab(self.tab_video, "🎬 视频分析")
        self.tabs.addTab(self.tab_feedback, "🔄 增量学习")
        self.tabs.addTab(self.tab_arch, "🏗️ 模型架构")

        main_layout.addWidget(self.tabs)

    def _init_statusbar(self):
        self.statusBar().showMessage(
            f"就绪 | 模型版本: v{self.model_version} | "
            f"反馈缓存: {self.feedback_buffer.size()}/{self.feedback_buffer.capacity}"
        )

    def _update_status(self):
        fb_count = self.feedback_buffer.size()
        fb_cap = self.feedback_buffer.capacity
        self.statusBar().showMessage(
            f"模型版本: v{self.model_version} | "
            f"反馈缓存: {fb_count}/{fb_cap} | "
            f"增量学习: {'运行中' if self.bg_updater and self.bg_updater.is_running() else '就绪'}"
        )
        self.tab_feedback.update_feedback_count(fb_count, fb_cap)

    def on_feedback(self, landmarks, label, is_correct):
        """处理用户反馈"""
        if landmarks is None:
            return

        if not is_correct:
            buf = list(self.feedback_frame_buffer)
            if len(buf) >= 8:
                features = self.preprocessor.process_sequence(buf)
                if features is not None:
                    self.feedback_buffer.add(features, label)
                    self.tab_feedback.train_log.append(
                        f"收到反馈: 纠正为「{ACTION_CLASSES[label]}」 "
                        f"(缓存: {self.feedback_buffer.size()}/{self.feedback_buffer.capacity})"
                    )
                    if self.feedback_buffer.is_full():
                        self.trigger_incremental_learning()
            else:
                self.tab_feedback.train_log.append("⚠ 反馈数据不足: 请在识别一段时间后再纠正")
        else:
            self.tab_feedback.train_log.append("收到反馈: 结果正确 ✓")

    def trigger_incremental_learning(self, force=False):
        if self.bg_updater and self.bg_updater.is_running():
            self.tab_feedback.train_log.append("⚠ 微调正在进行中, 请稍候...")
            return
        if not force and self.feedback_buffer.size() < 5:
            self.tab_feedback.train_log.append("⚠ 反馈数据不足 (至少需要5条)")
            return
        self.tab_feedback.train_log.append("🧠 触发EWC增量学习...")
        self.tab_feedback.learn_status_label.setText("🔄 正在微调...")
        
        def _bridge_progress(epoch, total, loss):
            self.train_progress_signal.emit(epoch, total, loss)
        def _bridge_complete(success, model_path):
            self.train_complete_signal.emit(success, model_path)
        
        self.bg_updater = BackgroundUpdater(
            self.feedback_buffer,
            on_complete=_bridge_complete,
            on_progress=_bridge_progress,
        )
        self.bg_updater.start()

    def _on_train_progress_safe(self, epoch, total, loss):
        self.tab_feedback.on_train_progress(epoch, total, loss)

    def _on_train_complete_safe(self, success, model_path):
        self.tab_feedback.on_train_complete(success)
        if success:
            self.model_version += 1
            self.tab_feedback.model_version_label.setText(
                f"v{self.model_version} (EWC更新)"
            )
            if hasattr(self.tab_realtime, 'video_thread') and \
               self.tab_realtime.video_thread and \
               self.tab_realtime.video_thread.engine:
                self.tab_realtime.video_thread.engine.hot_reload_dl_model(model_path)

    def closeEvent(self, event):
        if self.tab_realtime.video_thread is not None:
            self.tab_realtime.video_thread.stop()
        if hasattr(self.tab_video, 'analysis_thread') and \
           self.tab_video.analysis_thread is not None and \
           self.tab_video.analysis_thread.isRunning():
            self.tab_video.analysis_thread.stop()
            self.tab_video.analysis_thread.wait(3000)
        if self.bg_updater and self.bg_updater.is_running():
            self.bg_updater.join(timeout=5000)
        self.status_timer.stop()
        if self.feedback_buffer.size() > 0:
            self.feedback_buffer.save_to_disk()
        event.accept()


# ================================================================
# 程序入口
# ================================================================

def main():
    # 设置 matplotlib 中文字体
    if MATPLOTLIB_OK:
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        matplotlib.rcParams['axes.unicode_minus'] = False

    app = QApplication(sys.argv)
    app.setStyleSheet(GUI_STYLESHEET)

    # 设置全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    print("=" * 60)
    print("  基于深度学习的人体行为识别系统 (HARS)")
    print("  Human Action Recognition System")
    print("=" * 60)
    print("  系统已启动!")
    print("  • Tab 1: 实时识别 — 打开摄像头进行行为识别")
    print("  • Tab 2: 视频分析 — 上传视频文件分析")
    print("  • Tab 3: 增量学习 — 反馈纠正 & EWC微调")
    print("  • Tab 4: 模型架构 — 技术原理展示")
    print("=" * 60)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
