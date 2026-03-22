#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
游戏 AI 模型 - 深度学习核心
包含：状态编码器、UI 检测器、任务识别器、决策网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from pathlib import Path


class GameStateEncoder(nn.Module):
    """游戏状态编码器 (ResNet 风格 CNN)"""

    def __init__(self, image_size=(224, 224), embedding_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.state_embedding = nn.Linear(512, embedding_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        for i in range(blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3,
                         stride=stride if i == 0 else 1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return F.relu(self.state_embedding(x))


class UIElementDetector(nn.Module):
    """UI 元素检测器 (YOLO 风格)"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = GameStateEncoder(embedding_dim=256)
        self.detect_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.bbox_head = nn.Linear(64, 4)
        self.class_head = nn.Linear(64, num_classes)
        self.confidence_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.backbone(x)
        x = self.detect_head(features)
        return {
            "bbox": torch.sigmoid(self.bbox_head(x)),
            "class": self.class_head(x),
            "confidence": torch.sigmoid(self.confidence_head(x))
        }


class QuestRecognizer(nn.Module):
    """任务识别器"""

    def __init__(self, num_quest_types=20):
        super().__init__()
        self.visual_encoder = GameStateEncoder(embedding_dim=512)
        self.quest_classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_quest_types)
        )

    def forward(self, image):
        features = self.visual_encoder(image)
        return F.softmax(self.quest_classifier(features), dim=-1)


class ActionDecisionNetwork(nn.Module):
    """动作决策网络"""

    def __init__(self, state_dim=512, num_actions=50):
        super().__init__()
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        action_logits = self.policy_network(state)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.value_network(state)
        return {"action_probs": action_probs, "value": value}

    def select_action(self, state, deterministic=False):
        output = self.forward(state)
        action_probs = output["action_probs"]
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            prob = torch.max(action_probs, dim=-1).values
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            prob = dist.log_prob(action).exp()
        return action.item(), prob.item()


class GameAIModel:
    """游戏 AI 模型包装器"""

    def __init__(self, device="cpu", checkpoint_path=None):
        self.device = torch.device(device)
        self.state_encoder = GameStateEncoder().to(self.device)
        self.ui_detector = UIElementDetector().to(self.device)
        self.quest_recognizer = QuestRecognizer().to(self.device)
        self.decision_network = ActionDecisionNetwork().to(self.device)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
        
        self.action_map = {
            0: "click", 1: "double_click", 2: "swipe_up", 3: "swipe_down",
            4: "swipe_left", 5: "swipe_right", 6: "track_quest",
            7: "claim_reward", 8: "wait", 9: "break_loop",
        }

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if "state_encoder" in checkpoint:
            self.state_encoder.load_state_dict(checkpoint["state_encoder"])
        print(f"✓ 加载预训练模型：{path}")

    def save_checkpoint(self, path):
        checkpoint = {
            "state_encoder": self.state_encoder.state_dict(),
            "ui_detector": self.ui_detector.state_dict(),
            "quest_recognizer": self.quest_recognizer.state_dict(),
            "decision_network": self.decision_network.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"✓ 保存模型：{path}")

    def preprocess_image(self, image):
        import cv2
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def infer(self, screen_image):
        image_tensor = self.preprocess_image(screen_image)
        state = self.state_encoder(image_tensor)
        ui_elements = self.ui_detector(image_tensor)
        quest_probs = self.quest_recognizer(image_tensor)
        action_id, action_prob = self.decision_network.select_action(state, deterministic=True)
        
        return {
            "state_vector": state.cpu().detach().numpy(),
            "ui_elements": {k: v.cpu().detach().numpy() for k, v in ui_elements.items()},
            "quest_type": torch.argmax(quest_probs, dim=-1).item(),
            "action_id": action_id,
            "action_name": self.action_map.get(action_id, "unknown"),
            "action_probability": action_prob
        }


if __name__ == "__main__":
    print("GameAIModel 模块")
    print("=" * 60)
    model = GameStateEncoder()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"GameStateEncoder 参数量：{total_params:,}")
    
    ui_model = UIElementDetector()
    ui_params = sum(p.numel() for p in ui_model.parameters())
    print(f"UIElementDetector 参数量：{ui_params:,}")
    
    total = total_params + ui_params
    print(f"\n总参数量：~{total:,} ({total/1000000:.1f}M)")
