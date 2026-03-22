# 星穹铁道 AI 自动化系统

> ⚠️ **重要提示**: 这是一个**学习和研究用途**的项目。使用自动化程序可能违反游戏服务条款，请谨慎使用，后果自负。

## 🎯 项目特点

这不是传统的脚本！这是一个**真正的深度学习 AI 系统**：

| 传统脚本 | 本 AI 系统 |
|---------|-----------|
| 硬编码规则 | 深度学习理解游戏 |
| 无法适应变化 | 可以泛化到新情况 |
| 需要手动更新 | 可以自主学习 |
| 模板匹配 | 神经网络识别 |

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    星穹铁道 AI 系统                       │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ 屏幕捕获    │  │ 视觉理解    │  │ OCR 文字识别 │     │
│  │ Screen      │  │ CNN/ResNet  │  │ PaddleOCR   │     │
│  │ Capture     │  │ Encoder     │  │ + NLP       │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         └────────────────┼────────────────┘             │
│                          │                              │
│                  ┌───────▼────────┐                     │
│                  │  决策引擎      │                     │
│                  │  Decision      │                     │
│                  │  Network       │                     │
│                  └───────┬────────┘                     │
│                          │                              │
│         ┌────────────────┼────────────────┐             │
│         │                │                │             │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐     │
│  │ 模仿学习    │  │ PPO 强化学习 │  │ 动作执行    │     │
│  │ Imitation   │  │ Reinforcement│  │ Control     │     │
│  │ Learning    │  │ Learning    │  │             │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
starrail_ai/
├── ai_models/              # AI 模型核心
│   ├── game_ai_model.py    # 游戏理解深度学习模型
│   ├── smart_ocr.py        # 智能 OCR 文字识别
│   ├── decision_engine.py  # 智能决策引擎
│   └── reinforcement_learning.py  # 强化学习训练
├── core/                   # 核心功能模块
│   ├── screen_capture.py   # 屏幕捕获
│   └── image_recognition.py # 图像识别
├── config/                 # 配置文件
│   └── config.yaml         # AI 系统配置
├── assets/                 # 资源文件
│   ├── templates/          # 图像模板
│   └── screenshots/        # 截图保存
├── training/               # 训练相关
├── logs/                   # 日志目录
├── main.py                 # 主程序入口
├── pyproject.toml          # 项目依赖配置
└── README.md               # 项目说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install -e .

# 或者手动安装
pip install torch torchvision opencv-python numpy Pillow mss pyyaml
pip install paddlepaddle paddleocr  # OCR 功能
pip install pyautogui pygetwindow   # 自动化控制
```

### 2. 运行演示

```bash
cd starrail_ai
python main.py demo
```

### 3. 训练模型

```bash
# 模仿学习 (需要人类玩家演示数据)
python main.py train --mode imitation --data demonstrations.json

# PPO 强化学习
python main.py train --mode ppo
```

### 4. 运行自动化

```bash
# 日常任务模式
python main.py run --mode daily --duration 3600

# 指定设备 (GPU 加速)
python main.py run --device cuda
```

## 🧠 AI 模型详解

### 1. GameAIModel - 游戏理解模型

```python
# 状态编码器 (ResNet 风格 CNN)
GameStateEncoder:
  - 输入：224x224 游戏截图
  - 输出：512 维状态向量
  - 用途：将游戏画面编码为 AI 可理解的状态

# UI 元素检测器 (YOLO 风格)
UIElementDetector:
  - 输入：游戏截图
  - 输出：可交互元素位置和类型
  - 用途：识别按钮、图标等 UI 元素

# 任务识别器
QuestRecognizer:
  - 输入：游戏截图 + OCR 文字
  - 输出：任务类型概率分布
  - 用途：理解当前可执行的任务

# 决策网络
ActionDecisionNetwork:
  - 输入：状态向量
  - 输出：动作概率分布
  - 用途：决定下一步做什么
```

### 2. SmartOCR - 智能文字识别

```python
# 不只是识别文字，还能理解含义
SmartOCR:
  - 文字识别：PaddleOCR
  - 文字理解：关键词分类 + NLP
  - 任务解析：正则 + 语义理解
  
# 示例
识别："完成 1 次历战余烬"
理解：{
  "quest_type": "battle",
  "action": "fight",
  "target_count": 1
}
```

### 3. DecisionEngine - 智能决策

```python
# 基于多模态理解做决策
IntelligentDecisionEngine:
  1. 分析屏幕 (AI 视觉 + OCR)
  2. 分类界面类型
  3. 获取可用动作
  4. 计算优先级
  5. 输出决策
  
# 上下文感知
- 记录决策历史
- 检测死循环
- 避免重复动作
```

### 4. 强化学习训练

```python
# 模仿学习
ImitationLearning:
  - 记录人类玩家操作
  - 学习 (state, action) 映射
  - 快速获得基础策略

# PPO 强化学习
PPOAgent:
  - 试错学习
  - 奖励驱动优化
  - 超越人类策略
```

## 📊 训练数据收集

AI 需要训练数据才能工作。收集方法：

### 方法 1: 人工演示录制

```python
# 玩游戏时记录 (state, action) 对
from starrail_ai.main import StarRailAI

ai = StarRailAI()

# 每次你操作时记录
state = ai.get_state()  # 当前游戏状态
action = your_action   # 你的操作
ai.record_demonstration(state, action)
```

### 方法 2: 自动收集

```python
# 运行游戏时自动保存状态
ai.start_data_collection(output_path="training_data/")
```

## ⚙️ 配置说明

编辑 `config/config.yaml`:

```yaml
[general]
device = "cpu"  # 或 "cuda" 启用 GPU

[ai_model]
confidence_threshold = 0.7  # AI 置信度阈值

[automation]
action_delay = 0.5  # 动作延迟 (秒)
enable_random_delay = true  # 随机延迟 (防检测)

[daily_tasks]
daily_quest_priority = 10  # 每日任务优先级
auto_claim_reward = true   # 自动领奖

[safety]
max_continuous_actions = 100  # 最大连续操作
enable_loop_detection = true  # 死循环检测
```

## 🔬 技术细节

### 神经网络架构

```
GameStateEncoder:
  Conv2d(3, 64, k=7, s=2) → BatchNorm → ReLU
  ↓
  ResidualLayer1(64→64, 2 blocks)
  ↓
  ResidualLayer2(64→128, 2 blocks, stride=2)
  ↓
  ResidualLayer3(128→256, 2 blocks, stride=2)
  ↓
  ResidualLayer4(256→512, 2 blocks, stride=2)
  ↓
  AdaptiveAvgPool → Linear(512, 512) → ReLU
  ↓
  State Embedding (512-dim)
```

### 动作空间

AI 可以执行的动作 (由学习得到，不是硬编码):

| ID | 动作类型 | 说明 |
|----|---------|------|
| 0-9 | 点击类 | 单击、双击、长按等 |
| 10-19 | 滑动类 | 上下左右滑动 |
| 20-29 | 导航类 | 打开菜单、传送等 |
| 30-39 | 交互类 | 对话、领取、战斗等 |
| 40-49 | 保留 | 未来扩展 |

## ⚠️ 风险提示

1. **封号风险**: 使用自动化可能违反游戏服务条款
2. **学习用途**: 本项目主要用于学习 AI 技术
3. **自行负责**: 使用后果由用户自行承担

## 📚 学习资源

- **深度学习**: PyTorch 官方教程
- **强化学习**: [Spinning Up in Deep RL](https://spinningup.openai.com/)
- **计算机视觉**: [OpenCV 教程](https://docs.opencv.org/)
- **OCR 技术**: [PaddleOCR 文档](https://github.com/PaddlePaddle/PaddleOCR)

## 🤝 贡献

欢迎提交 Issue 和 PR!

## 📄 许可证

MIT License

---

**记住**: 这是一个 AI 学习项目，真正的智能需要大量训练数据和时间。不要期望它像魔法一样立即完美工作！
