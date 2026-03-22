# 星穹铁道 AI 自动化系统

> ⚠️ **重要提示**: 学习和研究用途，使用后果自负。

## 🎯 项目特点

**这不是传统的脚本！这是一个真正的深度学习 AI 系统：**

| 传统脚本 | 本 AI 系统 |
|---------|-----------|
| 硬编码规则 | 深度学习理解游戏 |
| 无法适应变化 | 可以泛化到新情况 |
| 需要手动更新 | 可以自主学习 |
| 模板匹配 | 神经网络识别 |

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/q602209037-png/test1.git
cd test1/starrail_ai
```

### 2. 安装依赖

```bash
pip install torch torchvision opencv-python numpy Pillow mss pyyaml
pip install paddlepaddle paddleocr pyautogui
```

### 3. 运行演示

```bash
python main.py demo
```

### 4. 运行自动化

```bash
# 日常任务模式
python main.py run --mode daily

# 指定运行时长 (3600 秒 = 1 小时)
python main.py run --mode daily --duration 3600

# 使用 GPU 加速
python main.py run --mode daily --device cuda
```

## 📁 项目结构

```
starrail_ai/
├── main.py                   # 主程序
├── ai_models/
│   ├── game_ai_model.py      # 深度学习模型 (~2.5M 参数)
│   ├── smart_ocr.py          # 智能 OCR 文字识别
│   └── decision_engine.py    # 智能决策引擎
├── core/
│   └── screen_capture.py     # 屏幕捕获模块
├── config/
│   └── config.yaml           # 配置文件
└── README.md
```

## 🧠 AI 架构

### 1. GameStateEncoder - 游戏状态编码器
- **类型**: ResNet 风格 CNN
- **参数量**: ~2.5M
- **功能**: 将游戏画面编码为 512 维状态向量

### 2. UIElementDetector - UI 元素检测器
- **类型**: YOLO 风格检测器
- **功能**: 检测界面上的可交互元素

### 3. QuestRecognizer - 任务识别器
- **功能**: 识别当前可执行的任务类型

### 4. ActionDecisionNetwork - 动作决策网络
- **功能**: 根据游戏状态决定下一步动作

### 5. SmartOCR - 智能 OCR
- **技术**: PaddleOCR + NLP
- **功能**: 识别并理解游戏文字

## 🔄 工作流程

1. **屏幕捕获** → 使用 MSS 库捕获游戏画面
2. **视觉理解** → CNN 编码为状态向量
3. **文字识别** → OCR 识别任务文本
4. **智能决策** → 神经网络选择最优动作
5. **执行动作** → PyAutoGUI 模拟操作
6. **反馈学习** → 记录 (state, action) 用于训练

## ⚠️ 风险提示

1. **封号风险**: 使用自动化可能违反游戏服务条款
2. **学习用途**: 本项目主要用于学习 AI 技术
3. **自行负责**: 使用后果由用户自行承担

## 📚 学习资源

- 深度学习：PyTorch 官方教程
- 强化学习：Spinning Up in Deep RL
- 计算机视觉：OpenCV 教程
- OCR 技术：PaddleOCR 文档

## 🤝 贡献

欢迎提交 Issue 和 PR!

## 📄 许可证

MIT License
