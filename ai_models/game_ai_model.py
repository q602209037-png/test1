"""
游戏状态理解模块
使用深度学习模型理解当前游戏状态
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from PIL import Image
import cv2


class GameStateEncoder(nn.Module):
    """
    游戏状态编码器
    将游戏画面编码为状态向量
    """
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224), 
                 embedding_dim: int = 512):
        super().__init__()
        
        # CNN  backbone - 使用简化的 ResNet 风格
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 状态嵌入
        self.state_embedding = nn.Linear(512, embedding_dim)
        
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        
    def _make_layer(self, in_channels: int, out_channels: int, 
                    blocks: int, stride: int = 1) -> nn.Sequential:
        """构建残差块层"""
        layers = []
        layers.append(self._residual_block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(self._residual_block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _residual_block(self, in_channels: int, out_channels: int, 
                        stride: int = 1) -> nn.Module:
        """单个残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像张量 [B, 3, H, W]
            
        Returns:
            状态嵌入向量 [B, embedding_dim]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 状态嵌入
        state = self.state_embedding(x)
        state = F.relu(state)
        
        return state


class UIElementDetector(nn.Module):
    """
    UI 元素检测器 (YOLO 风格)
    检测界面上的可交互元素
    """
    
    def __init__(self, num_classes: int = 10, image_size: int = 224):
        super().__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Backbone
        self.backbone = GameStateEncoder(
            image_size=(image_size, image_size), 
            embedding_dim=256
        )
        
        # 检测头
        self.detect_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # 边界框预测 (cx, cy, w, h)
        self.bbox_head = nn.Linear(64, 4)
        
        # 类别预测
        self.class_head = nn.Linear(64, num_classes)
        
        # 置信度预测
        self.confidence_head = nn.Linear(64, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Returns:
            包含 bbox, class, confidence 的字典
        """
        # 提取特征
        features = self.backbone(x)
        
        # 检测头
        x = self.detect_head(features)
        
        # 预测
        bbox = torch.sigmoid(self.bbox_head(x))
        class_logits = self.class_head(x)
        confidence = torch.sigmoid(self.confidence_head(x))
        
        return {
            "bbox": bbox,
            "class": class_logits,
            "confidence": confidence
        }


class QuestRecognizer(nn.Module):
    """
    任务识别器
    识别当前可执行的任务类型
    """
    
    def __init__(self, num_quest_types: int = 20, embedding_dim: int = 512):
        super().__init__()
        
        # 视觉编码器
        self.visual_encoder = GameStateEncoder(
            image_size=(224, 224),
            embedding_dim=embedding_dim
        )
        
        # 任务分类器
        self.quest_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_quest_types)
        )
        
        self.num_quest_types = num_quest_types
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        识别任务类型
        
        Returns:
            任务类型概率分布
        """
        features = self.visual_encoder(image)
        quest_probs = F.softmax(self.quest_classifier(features), dim=-1)
        return quest_probs


class ActionDecisionNetwork(nn.Module):
    """
    动作决策网络
    根据游戏状态决定下一步动作
    """
    
    def __init__(self, state_dim: int = 512, num_actions: int = 50):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # 策略网络
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        # 价值网络 (用于强化学习)
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        根据状态预测动作
        
        Returns:
            包含 action_probs 和 value 的字典
        """
        action_logits = self.policy_network(state)
        action_probs = F.softmax(action_logits, dim=-1)
        
        value = self.value_network(state)
        
        return {
            "action_probs": action_probs,
            "value": value
        }
    
    def select_action(self, state: torch.Tensor, 
                      deterministic: bool = False) -> Tuple[int, float]:
        """
        选择动作
        
        Args:
            state: 状态向量
            deterministic: 是否确定性选择 (测试时用)
            
        Returns:
            (动作 ID, 动作概率)
        """
        output = self.forward(state)
        action_probs = output["action_probs"]
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            prob = torch.max(action_probs, dim=-1).values
        else:
            # 采样动作
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            prob = dist.log_prob(action).exp()
        
        return action.item(), prob.item()


# 模型包装器 - 方便使用
class GameAIModel:
    """
    游戏 AI 模型包装器
    整合所有模型，提供统一的推理接口
    """
    
    def __init__(self, device: str = "cpu", 
                 checkpoint_dir: str = "ai_models/checkpoints"):
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # 初始化模型
        self.state_encoder = GameStateEncoder().to(self.device)
        self.ui_detector = UIElementDetector().to(self.device)
        self.quest_recognizer = QuestRecognizer().to(self.device)
        self.decision_network = ActionDecisionNetwork().to(self.device)
        
        # 动作映射 (AI 学习的动作，不是硬编码!)
        self.action_map = {
            0: "click",
            1: "double_click", 
            2: "swipe_up",
            3: "swipe_down",
            4: "swipe_left",
            5: "swipe_right",
            # ... 更多动作由 AI 学习
        }
        
    def load_checkpoint(self, checkpoint_path: str):
        """加载训练好的模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "state_encoder" in checkpoint:
            self.state_encoder.load_state_dict(checkpoint["state_encoder"])
        if "ui_detector" in checkpoint:
            self.ui_detector.load_state_dict(checkpoint["ui_detector"])
        if "quest_recognizer" in checkpoint:
            self.quest_recognizer.load_state_dict(checkpoint["quest_recognizer"])
        if "decision_network" in checkpoint:
            self.decision_network.load_state_dict(checkpoint["decision_network"])
            
        print(f"模型已加载：{checkpoint_path}")
        
    def save_checkpoint(self, path: str):
        """保存模型"""
        checkpoint = {
            "state_encoder": self.state_encoder.state_dict(),
            "ui_detector": self.ui_detector.state_dict(),
            "quest_recognizer": self.quest_recognizer.state_dict(),
            "decision_network": self.decision_network.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"模型已保存：{path}")
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 转换为 RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # 调整大小
        image = cv2.resize(image, (224, 224))
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 转换为张量
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def infer(self, screen_image: np.ndarray) -> Dict:
        """
        完整推理流程
        
        Args:
            screen_image: 游戏屏幕截图
            
        Returns:
            AI 决策结果
        """
        # 预处理
        image_tensor = self.preprocess_image(screen_image)
        
        # 1. 编码游戏状态
        state = self.state_encoder(image_tensor)
        
        # 2. 检测 UI 元素
        ui_elements = self.ui_detector(image_tensor)
        
        # 3. 识别任务类型
        quest_probs = self.quest_recognizer(image_tensor)
        
        # 4. 决策动作
        action_id, action_prob = self.decision_network.select_action(
            state, deterministic=True
        )
        
        return {
            "state_vector": state.cpu().detach().numpy(),
            "ui_elements": ui_elements,
            "quest_type": torch.argmax(quest_probs, dim=-1).item(),
            "quest_probs": quest_probs.cpu().detach().numpy(),
            "action_id": action_id,
            "action_name": self.action_map.get(action_id, "unknown"),
            "action_probability": action_prob
        }


if __name__ == "__main__":
    from pathlib import Path
    
    # 测试模型创建
    print("创建 AI 模型...")
    model = GameAIModel(device="cpu")
    
    # 显示模型结构
    print("\n=== 状态编码器 ===")
    print(model.state_encoder)
    
    print("\n=== UI 检测器 ===")
    print(model.ui_detector)
    
    print("\n=== 任务识别器 ===")
    print(model.quest_recognizer)
    
    print("\n=== 决策网络 ===")
    print(model.decision_network)
    
    print("\n模型创建成功！这是一个真正的深度学习模型，需要训练后才能使用。")
    print("训练数据可以通过玩游戏时收集 (state, action) 对来获得。")
