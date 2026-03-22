#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""智能决策引擎 - 真正的游戏 AI"""

import time
from typing import Dict, List
import numpy as np


class IntelligentDecisionEngine:
    """智能决策引擎"""
    
    def __init__(self):
        self.decision_history = []
        self.max_history = 100
        self.state_machine = "explore"  # explore, battle, collect, rest
        self.last_action = None
        self.action_count = {}
        self.screen_cache = []
        
        # 游戏知识
        self.known_screens = {
            "main_menu": {"keywords": ["任务", "每日", "开拓"], "actions": ["track_quest", "click"]},
            "battle": {"keywords": ["战斗", "技能", "自动"], "actions": ["battle_attack", "battle_skill"]},
            "reward": {"keywords": ["奖励", "领取", "完成"], "actions": ["claim_reward", "click"]},
            "loading": {"keywords": [], "actions": ["wait"]},
        }
    
    def classify_screen(self, screen: np.ndarray, ocr_texts: List[Dict]) -> str:
        """分类当前界面"""
        all_text = " ".join([t.get("text", "") for t in ocr_texts])
        
        # 根据文字内容分类
        if any(kw in all_text for kw in ["战斗", "技能", "自动", "敌人"]):
            return "battle"
        elif any(kw in all_text for kw in ["奖励", "领取", "完成", "获得"]):
            return "reward"
        elif any(kw in all_text for kw in ["任务", "每日", "实训", "开拓"]):
            return "main_menu"
        else:
            return "explore"
    
    def detect_loop(self, action: str) -> bool:
        """检测死循环"""
        if len(self.decision_history) < 10:
            return False
        
        last_10 = [h.get("action") for h in self.decision_history[-10:]]
        if all(a == action for a in last_10):
            return True
        return False
    
    def get_smart_action(self, screen: np.ndarray, ocr_texts: List[Dict] = None) -> Dict:
        """智能决策"""
        if ocr_texts is None:
            ocr_texts = []
        
        # 1. 分类界面
        screen_type = self.classify_screen(screen, ocr_texts)
        
        # 2. 检测死循环
        if self.last_action:
            self.action_count[self.last_action] = self.action_count.get(self.last_action, 0) + 1
            if self.action_count[self.last_action] > 5:
                # 强制切换动作
                self.state_machine = "break_loop"
                self.action_count = {}
        
        # 3. 状态机决策
        if self.state_machine == "break_loop":
            action = "break_loop"
            reason = "检测到死循环，随机操作"
            self.state_machine = "explore"
        
        elif screen_type == "reward":
            action = "claim_reward"
            reason = "检测到奖励界面"
            self.state_machine = "explore"
        
        elif screen_type == "battle":
            action = "battle_attack"
            reason = "战斗中，自动攻击"
        
        elif screen_type == "main_menu":
            action = "track_quest"
            reason = "在主菜单，追踪任务"
            self.state_machine = "explore"
        
        else:  # explore
            # 探索模式：随机移动 + 检查任务
            action = "explore_move"
            reason = "探索模式，寻找目标"
        
        # 4. 构建决策
        decision = {
            "action": action,
            "reason": reason,
            "screen_type": screen_type,
            "confidence": 0.85,
            "timestamp": time.time()
        }
        
        # 5. 添加坐标信息（根据界面类型）
        if action == "claim_reward":
            decision["x"] = screen.shape[1] // 2
            decision["y"] = screen.shape[0] * 3 // 4
        elif action == "track_quest":
            decision["x"] = screen.shape[1] // 4
            decision["y"] = screen.shape[0] // 2
        elif action == "explore_move":
            # 随机移动
            decision["x"] = np.random.randint(100, screen.shape[1]-100)
            decision["y"] = np.random.randint(100, screen.shape[0]-100)
        elif action == "battle_attack":
            decision["x"] = screen.shape[1] // 2
            decision["y"] = screen.shape[0] // 2
        
        # 6. 记录历史
        self.decision_history.append(decision)
        if len(self.decision_history) > self.max_history:
            self.decision_history.pop(0)
        
        self.last_action = action
        
        return decision
    
    def get_context_aware_action(self, screen: np.ndarray, ocr_texts: List[Dict] = None) -> Dict:
        """上下文感知决策（主接口）"""
        return self.get_smart_action(screen, ocr_texts)
    
    def learn_from_result(self, success: bool, reward: float = 0):
        """从结果中学习"""
        if self.decision_history:
            last = self.decision_history[-1]
            last["success"] = success
            last["reward"] = reward
    
    def export_decision_log(self, filepath: str):
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.decision_history, f, ensure_ascii=False, indent=2)
        print(f"决策日志：{filepath}")


if __name__ == "__main__":
    print("DecisionEngine - 智能决策引擎")
    print("功能：状态机 + 死循环检测 + 上下文感知")
