#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""智能 OCR 模块 - PaddleOCR + NLP 文字理解"""

from typing import List, Dict
import re

class SmartOCR:
    """智能 OCR 识别器"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.ocr = None
        self._init_ocr()
        self.quest_keywords = {
            "daily": ["每日", "日常", "实训", "活跃度"],
            "battle": ["战斗", "敌人", "副本"],
            "explore": ["探索", "宝箱", "解谜"],
            "npc": ["对话", "NPC"],
            "collect": ["收集", "领取", "奖励"],
        }

    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=self.use_gpu, show_log=False)
            print("PaddleOCR 初始化成功")
        except ImportError:
            print("警告：PaddleOCR 未安装，OCR 功能将不可用")

    def recognize(self, image) -> List[Dict]:
        if self.ocr is None:
            return []
        try:
            result = self.ocr.ocr(image, cls=True)
            if result is None or result[0] is None:
                return []
            texts = []
            for line in result[0]:
                if line:
                    bbox, (text, confidence) = line
                    texts.append({"text": text, "confidence": confidence, "bbox": bbox})
            return texts
        except Exception as e:
            print(f"OCR 识别失败：{e}")
            return []

    def understand_text(self, texts: List[Dict]) -> Dict:
        understanding = {
            "quest_type": None,
            "action_hint": None,
            "raw_texts": [t["text"] for t in texts]
        }
        all_text = " ".join([t["text"] for t in texts])
        for quest_type, keywords in self.quest_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    understanding["quest_type"] = quest_type
                    break
            if understanding["quest_type"]:
                break
        return understanding


class TextUnderstandingAI:
    """文本理解 AI"""
    
    def __init__(self):
        self.quest_patterns = [
            r"完成\s*(\d+)\s*次.*?(?:战斗 | 挑战)",
            r"消耗\s*(\d+)\s*点\s*(?:体力 | 开拓力)",
            r"与\s*(.+?)\s*对话",
            r"前往\s*(.+?)",
        ]

    def parse_quest_description(self, text: str) -> Dict:
        result = {"original_text": text, "quest_type": "unknown", "target_count": None, "action": None}
        for pattern in self.quest_patterns:
            match = re.search(pattern, text)
            if match:
                if "战斗" in text or "挑战" in text:
                    result["quest_type"] = "battle"
                    result["action"] = "fight"
                elif "对话" in text:
                    result["quest_type"] = "talk"
                    result["action"] = "talk"
                elif "前往" in text:
                    result["quest_type"] = "navigate"
                    result["action"] = "navigate"
                groups = match.groups()
                if groups and groups[0].isdigit():
                    result["target_count"] = int(groups[0])
                break
        return result

if __name__ == "__main__":
    print("SmartOCR 模块 - OCR 识别 + NLP 文字理解")
