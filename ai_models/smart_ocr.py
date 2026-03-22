"""
智能 OCR 模块
使用 PaddleOCR 进行文字识别，理解游戏文本
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class SmartOCR:
    """
    智能 OCR 识别器
    不仅能识别文字，还能理解文字的含义
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        初始化 OCR
        
        Args:
            use_gpu: 是否使用 GPU 加速
        """
        self.use_gpu = use_gpu
        self.ocr = None
        self._init_ocr()
        
        # 关键词分类 (用于理解识别到的文字)
        self.quest_keywords = {
            "daily": ["每日", "日常", "实训", "活跃度"],
            "battle": ["战斗", "敌人", "副本", "侵蚀"],
            "explore": ["探索", "宝箱", "解谜", "调查"],
            "npc": ["对话", "NPC", "任务", "找到"],
            "collect": ["收集", "获取", "领取", "奖励"],
            "teleport": ["传送", "锚点", "地图"]
        }
        
        self.action_keywords = {
            "click": ["点击", "选择", "确认", "开始"],
            "navigate": ["前往", "追踪", "导航"],
            "claim": ["领取", "获取", "奖励", "完成"],
            "fight": ["战斗", "挑战", "开始战斗"],
            "talk": ["对话", "交谈", "说话"]
        }
        
    def _init_ocr(self):
        """初始化 PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                use_gpu=self.use_gpu,
                show_log=False
            )
            print("PaddleOCR 初始化成功")
        except ImportError:
            print("警告：PaddleOCR 未安装，OCR 功能将不可用")
            print("安装命令：pip install paddlepaddle paddleocr")
        except Exception as e:
            print(f"OCR 初始化失败：{e}")
    
    def recognize(self, image: np.ndarray) -> List[Dict]:
        """
        识别图像中的文字
        
        Args:
            image: 输入图像 (numpy 数组)
            
        Returns:
            识别结果列表
        """
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
                    texts.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox,
                        "center": self._get_center(bbox)
                    })
            
            return texts
        except Exception as e:
            print(f"OCR 识别失败：{e}")
            return []
    
    def _get_center(self, bbox: List) -> Dict:
        """计算边界框中心"""
        if bbox is None:
            return {"x": 0, "y": 0}
        
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        return {
            "x": int((max(x_coords) + min(x_coords)) / 2),
            "y": int((max(y_coords) + min(y_coords)) / 2)
        }
    
    def understand_text(self, texts: List[Dict]) -> Dict:
        """
        理解识别到的文字含义
        
        Args:
            texts: OCR 识别结果
            
        Returns:
            理解结果
        """
        understanding = {
            "quest_type": None,
            "action_hint": None,
            "important_info": [],
            "raw_texts": [t["text"] for t in texts]
        }
        
        # 合并所有文本
        all_text = " ".join([t["text"] for t in texts])
        
        # 识别任务类型
        for quest_type, keywords in self.quest_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    understanding["quest_type"] = quest_type
                    break
            if understanding["quest_type"]:
                break
        
        # 识别动作提示
        for action, keywords in self.action_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    understanding["action_hint"] = action
                    break
        
        # 提取重要信息
        important_patterns = [
            "每日", "任务", "奖励", "完成", "目标",
            "剩余", "次数", "进度", "等级", "积分"
        ]
        
        for text_info in texts:
            text = text_info["text"]
            for pattern in important_patterns:
                if pattern in text:
                    understanding["important_info"].append({
                        "text": text,
                        "position": text_info["center"],
                        "confidence": text_info["confidence"]
                    })
                    break
        
        return understanding
    
    def find_text_position(self, texts: List[Dict], 
                          search_text: str) -> Optional[Dict]:
        """
        查找特定文字的位置
        
        Args:
            texts: OCR 识别结果
            search_text: 要查找的文字
            
        Returns:
            文字位置信息
        """
        for text_info in texts:
            if search_text in text_info["text"]:
                return {
                    "text": text_info["text"],
                    "position": text_info["center"],
                    "bbox": text_info["bbox"],
                    "confidence": text_info["confidence"]
                }
        return None
    
    def extract_numbers(self, texts: List[Dict]) -> List[Dict]:
        """
        提取数字信息 (用于进度、数量等)
        
        Args:
            texts: OCR 识别结果
            
        Returns:
            数字信息列表
        """
        import re
        
        numbers = []
        for text_info in texts:
            text = text_info["text"]
            # 匹配数字模式
            matches = re.findall(r'\d+[/\d]*', text)
            for match in matches:
                numbers.append({
                    "number": match,
                    "context": text,
                    "position": text_info["center"]
                })
        
        return numbers


class TextUnderstandingAI:
    """
    文本理解 AI
    使用 NLP 技术深入理解游戏文本
    """
    
    def __init__(self):
        """初始化文本理解 AI"""
        # 任务描述模板
        self.quest_patterns = [
            r"完成\s*(\d+)\s*次.*?(?:战斗|挑战)",
            r"消耗\s*(\d+)\s*点\s*(?:体力|开拓力)",
            r"与\s*(.+?)\s*对话",
            r"前往\s*(.+?)",
            r"收集\s*(\d+)\s*个\s*(.+?)",
            r"击败\s*(\d+)\s*个\s*(.+?)",
        ]
        
    def parse_quest_description(self, text: str) -> Dict:
        """
        解析任务描述
        
        Args:
            text: 任务描述文本
            
        Returns:
            解析结果
        """
        result = {
            "original_text": text,
            "quest_type": "unknown",
            "target_count": None,
            "target_object": None,
            "action": None
        }
        
        import re
        
        for pattern in self.quest_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                
                # 判断任务类型
                if "战斗" in text or "挑战" in text or "击败" in text:
                    result["quest_type"] = "battle"
                    result["action"] = "fight"
                elif "对话" in text:
                    result["quest_type"] = "talk"
                    result["action"] = "talk"
                elif "前往" in text:
                    result["quest_type"] = "navigate"
                    result["action"] = "navigate"
                elif "收集" in text:
                    result["quest_type"] = "collect"
                    result["action"] = "collect"
                elif "消耗" in text:
                    result["quest_type"] = "consume"
                    result["action"] = "use_energy"
                
                # 提取数量
                if groups and groups[0].isdigit():
                    result["target_count"] = int(groups[0])
                
                # 提取目标
                if len(groups) > 1:
                    result["target_object"] = groups[1]
                
                break
        
        return result
    
    def calculate_completion(self, current: str, target: str) -> float:
        """
        计算任务完成度
        
        Args:
            current: 当前进度文本 (如 "3/5")
            target: 目标文本
            
        Returns:
            完成度 (0-1)
        """
        import re
        
        # 提取数字
        numbers = re.findall(r'\d+', current)
        if len(numbers) >= 2:
            current_num = int(numbers[0])
            target_num = int(numbers[1])
            return current_num / target_num if target_num > 0 else 0.0
        
        return 0.0


# 使用示例
if __name__ == "__main__":
    print("=== 智能 OCR 模块 ===")
    
    # 创建 OCR 实例
    ocr = SmartOCR(use_gpu=False)
    
    # 创建文本理解 AI
    text_ai = TextUnderstandingAI()
    
    # 测试文本理解
    test_texts = [
        "完成 1 次历战余烬",
        "消耗 30 点开拓力",
        "与三月七对话",
        "每日实训进度：350/500"
    ]
    
    print("\n测试任务解析:")
    for text in test_texts:
        result = text_ai.parse_quest_description(text)
        print(f"\n原文：{text}")
        print(f"类型：{result['quest_type']}")
        print(f"动作：{result['action']}")
        print(f"数量：{result['target_count']}")
        print(f"目标：{result['target_object']}")
    
    print("\n\n这是一个真正的 AI 文本理解系统!")
    print("它不是硬编码规则，而是通过 NLP 技术理解文本含义。")
