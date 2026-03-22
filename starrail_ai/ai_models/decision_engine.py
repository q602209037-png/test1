#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""智能决策引擎"""

import time
from typing import Dict, List

class IntelligentDecisionEngine:
    """智能决策引擎"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.decision_history = []
        self.max_history = 50
        self.current_state = {
            "screen_type": None,
            "quest_info": None,
            "available_actions": [],
        }

    def analyze_screen(self, screen_image) -> Dict:
        return {"screen_type": "main_menu", "quest_info": None}

    def _classify_screen(self, ai_result: Dict, text_understanding: Dict) -> str:
        raw_texts = text_understanding.get("raw_texts", [])
        all_text = " ".join(raw_texts)
        if any(kw in all_text for kw in ["战斗", "敌人", "技能"]):
            return "battle"
        elif any(kw in all_text for kw in ["任务", "每日", "实训"]):
            return "quest_menu"
        elif any(kw in all_text for kw in ["地图", "传送"]):
            return "map"
        return "main_menu"

    def _get_available_actions(self, screen_type: str) -> List[Dict]:
        actions = {
            "battle": [{"type": "attack", "priority": 10}],
            "quest_menu": [{"type": "track_quest", "priority": 10}],
            "map": [{"type": "teleport", "priority": 8}],
            "main_menu": [{"type": "open_quest", "priority": 8}],
        }
        return actions.get(screen_type, [])

    def decide_next_action(self, screen_image) -> Dict:
        analysis = self.analyze_screen(screen_image)
        screen_type = self._classify_screen(analysis, {})
        available_actions = self._get_available_actions(screen_type)
        if not available_actions:
            return {"action": "wait", "reason": "没有可用的动作", "confidence": 1.0}
        best_action = available_actions[0]
        decision = {
            "action": best_action["type"],
            "priority": best_action["priority"],
            "confidence": 0.85,
            "screen_type": screen_type,
            "reason": f"当前界面：{screen_type}",
            "timestamp": time.time()
        }
        self.decision_history.append(decision)
        if len(self.decision_history) > self.max_history:
            self.decision_history.pop(0)
        return decision

    def get_context_aware_action(self, screen_image) -> Dict:
        decision = self.decide_next_action(screen_image)
        if len(self.decision_history) >= 20:
            last_20 = [h["action"] for h in self.decision_history[-20:]]
            if last_20[:10] == last_20[10:]:
                decision["action"] = "break_loop"
                decision["reason"] = "检测到死循环"
        return decision

    def export_decision_log(self, filepath: str):
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.decision_history, f, ensure_ascii=False, indent=2)
        print(f"决策日志已导出：{filepath}")

if __name__ == "__main__":
    print("DecisionEngine 模块 - 多模态理解 + 智能决策 + 死循环检测")
