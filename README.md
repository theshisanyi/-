# 🧠 基于深度学习的人体行为识别系统 (HARS)

> **Human Action Recognition System** — 基于 ST-GCN + Transformer 混合模型的实时人体行为识别

---

## 📋 项目概述

本系统是一个完整的人体行为识别解决方案，可从实时视频流或离线视频文件中识别 **10 种人体动作**：

| 编号 | 中文 | English | 核心特征 |
|------|------|---------|----------|
| 0 | 行走 | Walking | 脚步交替 + 中等速度 |
| 1 | 跑步 | Running | 高频步伐 + 大幅摆臂 |
| 2 | 跳跃 | Jumping | 重心突变 + 膝关节弯曲 |
| 3 | 坐下 | Sitting | 髋膝角减小 + 低运动 |
| 4 | 站立 | Standing | 直立 + 静止 |
| 5 | 挥手 | Waving | 手高于肩 + 水平振荡 |
| 6 | 弯腰 | Bending | 躯干前倾 + 髋角减小 |
| 7 | 举手 | Raising Hand | 手腕高于头部 |
| 8 | 踢腿 | Kicking | 单腿高速运动 |
| 9 | 跌倒 | Falling | 重心急降 + 躯干倾斜 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    PyQt5 GUI 主界面                      │
│  ┌──────────┬──────────┬──────────┬──────────┐          │
│  │ 实时识别  │ 视频分析  │ 增量学习  │ 模型架构  │          │
│  └──────────┴──────────┴──────────┴──────────┘          │
├─────────────────────────────────────────────────────────┤
│                   推理引擎 (三模式)                       │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────┐       │
│  │ 规则分类器    │ │ ONNX推理   │ │ 混合模式     │       │
│  └──────────────┘ └────────────┘ └──────────────┘       │
├─────────────────────────────────────────────────────────┤
│               数据处理 Pipeline                          │
│  MediaPipe → 归一化 → 速度特征 → 自适应采样               │
├─────────────────────────────────────────────────────────┤
│               深度学习模型                                │
│  多流嵌入 → ST-GCN×2 → Transformer×3 → 分类头           │
├─────────────────────────────────────────────────────────┤
│               增量学习 (EWC)                              │
│  反馈缓冲 → Fisher矩阵 → EWC微调 → 模型热更新            │
└─────────────────────────────────────────────────────────┘
```



### 核心创新点

1. **ST-GCN + Transformer 混合模型** — 自适应图卷积 + 多头自注意力
2. **时序自适应采样** — 根据运动量动态调整处理帧数
3. **EWC 增量学习** — 在线微调 + 灾难性遗忘防护
4. **模型压缩** — 结构化剪枝 (30%) + INT8 量化感知训练
5. **双模式推理** — 规则分类器 (即时可用) + 深度学习模型

---

## 📁 项目结构

```
zzqzz/
├── main_system.py          # 🎯 PyQt5 GUI 主程序 (程序入口)
├── config.py               # ⚙️ 全局配置 (动作类别、超参数、路径)
├── utils.py                # 🛠️ 工具函数 (角度计算、骨骼绘制、FPS)
├── data_preprocessor.py    # 📊 数据预处理 (MediaPipe + 归一化 + 速度)
├── model.py                # 🧠 ST-GCN + Transformer 混合模型
├── rule_classifier.py      # 📐 规则分类器 (几何特征识别)
├── inference_engine.py     # ⚡ 推理引擎 (规则/DL/混合 三模式)
├── incremental_learner.py  # 🔄 增量学习 (EWC + 后台微调)
├── train.py                # 📈 全监督训练脚本
├── pretrain.py             # 🔬 SimCLR 自监督预训练
├── optimize_model.py       # ✂️ 模型压缩 (剪枝 + 量化 + ONNX导出)
├── generate_demo_model.py  # 🎮 生成演示用ONNX模型
├── requirements.txt        # 📦 Python 依赖
├── models/                 # 模型文件目录 (自动创建)
├── feedback_data/          # 反馈数据目录 (自动创建)
└── logs/                   # 训练日志目录 (自动创建)
```

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.8 及以上
- **操作系统**: Windows / macOS / Linux
- **硬件**: 
  - 推理: 标准 CPU (Intel i5 级别即可)
  - 训练: 推荐 NVIDIA GPU (CUDA 11.0+)
  - 摄像头: 用于实时识别

### 第一步: 安装依赖

```bash
cd zzqzz
pip install -r requirements.txt
```

> **依赖列表**:
> - `torch` (>=1.10): 深度学习框架
> - `onnxruntime` (>=1.12): ONNX 推理
> - `mediapipe` (>=0.10): 姿态估计
> - `opencv-python` (>=4.5): 视频处理
> - `PyQt5` (>=5.15): GUI 界面
> - `scipy`: 科学计算
> - `matplotlib`: 数据可视化
> - `numpy`: 数值计算

### 第二步: 启动系统 (规则模式 — 无需训练)

```bash
python main_system.py
```

系统启动后，在界面中选择 **「规则模式」** 即可立即使用：

1. 点击 **「▶ 开始识别」** 打开摄像头
2. 在镜头前做出动作，系统自动识别
3. 可切换 **「规则模式」/「深度学习模式」/「混合模式」**

### 第三步 (可选): 训练深度学习模型

```bash
# 1. 生成演示用ONNX模型 (随机权重, 用于快速测试DL模式)
python generate_demo_model.py

# 2. 或者使用合成数据进行训练 (推荐)
python train.py

# 3. 训练后模型压缩与ONNX导出
python optimize_model.py
```

### 第四步 (可选): 自监督预训练

```bash
# SimCLR 对比学习预训练 (可提升最终模型精度)
python pretrain.py
```

---

## 📦 部署到另一台电脑 (打包与迁移)

如果你需要将此项目在另一台电脑上运行 (例如用于毕业答辩)，请严格按照以下步骤操作以确保不会出现环境或缺少模型的问题：

### 1. 复制项目文件
将整个 `zzqzz` 文件夹原封不动地通过 U 盘或网盘复制到目标电脑上。
**注意**：目标电脑需要自行安装 Python。请确保复制时包含了以下自动生成的目录：
- `models/` (内含 `hars_model.onnx`, `hars_model.pt` 和自动下载的 `pose_landmarker_lite.task`)
- `feedback_data/` (如果已有增量学习数据)

### 2. 目标电脑环境准备
在目标电脑上安装 **Python 3.8 ~ 3.11** 之间的版本（推荐 3.10 或 3.11），安装时**务必勾选 "Add python.exe to PATH"**。

### 3. 一键安装依赖
在目标电脑的命令行终端（CMD 或 PowerShell）中，进入项目所在目录并执行：
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
*(注：`requirements.txt` 中已包含项目严格的依赖列表，运行完全版所需的一切库都已经就绪)*

### 4. 验证并运行
在目标电脑上，直接运行主系统即可：
```bash
python main_system.py
```
> **安全提示**：如果你在答辩时没有摄像头设备，GUI 依然能够正常启动，控制台输出 `[错误] 无法打开视频源: 0` 是正常现象。此时在 GUI 的「视频分析」Tab 页面中，直接上传测试用的视频文件即可向老师演示你的系统。

---

## 🖥️ GUI 操作指南

### Tab 1: 实时识别
- 打开摄像头实时识别人体动作
- 支持三种推理模式切换
- 实时显示骨架、动作标签、置信度、FPS
- 10 类行为概率分布可视化

### Tab 2: 视频分析
- 上传视频文件 (MP4/AVI/MOV/MKV)
- 自动逐帧分析行为
- 生成行为分布统计报告和饼图

### Tab 3: 增量学习
- 当识别结果错误时，点击「✗ 错误」并选择正确标签
- 反馈数据自动缓存
- 缓冲区满后可触发 EWC 增量微调
- 微调完成后自动热更新模型

### Tab 4: 模型架构
- 展示 ST-GCN + Transformer 混合模型架构图
- MediaPipe 33 关键点分布
- 技术原理和创新点详细说明

---

## 🔧 配置说明

所有配置集中在 `config.py`：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `MODEL_CONFIG.embed_dim` | 嵌入维度 | 128 |
| `MODEL_CONFIG.num_stgcn_layers` | ST-GCN 层数 | 2 |
| `MODEL_CONFIG.num_transformer_layers` | Transformer 层数 | 3 |
| `MODEL_CONFIG.num_heads` | 注意力头数 | 8 |
| `TRAIN_CONFIG.epochs` | 训练轮数 | 100 |
| `TRAIN_CONFIG.batch_size` | 批大小 | 32 |
| `EWC_CONFIG.ewc_lambda` | EWC 正则化强度 | 5000 |
| `EWC_CONFIG.buffer_capacity` | 反馈缓冲区容量 | 100 |
| `INFERENCE_CONFIG.confidence_threshold` | 置信度阈值 | 0.3 |
| `SAMPLING_CONFIG.max_frames` | 最大采样帧数 | 64 |

---

## 📊 技术细节

### 模型架构

```
输入 (B, T, 33, 6) → 多流嵌入 (128d)
  → ST-GCN ×2 (自适应图卷积 + 时序卷积)
  → 空间池化 → 位置编码
  → Transformer ×3 (8头自注意力)
  → 全局平均池化 → 分类头 → 10类 Softmax
```

### 规则分类器特征

基于 MediaPipe 33 关键点的几何特征：
- **关节角度**: 肘关节角、膝关节角、髋关节角
- **身体倾斜**: 肩膀-髋部连线与垂直线夹角
- **运动速度**: 帧间关键点位移
- **相对位置**: 手腕与肩膀/头部的高度比较

### EWC 增量学习

$$L_{total} = L_{task} + \lambda \sum_i F_i (\theta_i - \theta^*_i)^2$$

- $F_i$: Fisher 信息矩阵 (参数重要性)
- $\theta^*_i$: 旧任务最优参数
- $\lambda$: 正则化强度 (默认 5000)

---

## ⚠️ 常见问题

### 1. MediaPipe 安装失败
```bash
pip install mediapipe --no-cache-dir
```

### 2. PyQt5 显示异常
```bash
pip install PyQt5==5.15.9
```

### 3. 摄像头无法打开
- 检查摄像头是否被其他程序占用
- 尝试修改 `main_system.py` 中的 `source=0` 为 `source=1`

### 4. ONNX 模型不存在
```bash
python generate_demo_model.py  # 生成演示模型
```

---

## 📝 License

本项目为毕业设计作品，仅供学术研究和学习交流使用。
