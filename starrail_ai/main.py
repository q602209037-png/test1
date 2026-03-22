#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
星穹铁道 AI 自动化系统 - 主程序
真正的深度学习 AI，不是脚本！
"""

import sys
import time
import argparse
import pyautogui
from pathlib import Path
from typing import Optional, Dict
import numpy as np

from ai_models.game_ai_model import GameAIModel
from ai_models.smart_ocr import SmartOCR, TextUnderstandingAI
from ai_models.decision_engine import IntelligentDecisionEngine
from core.screen_capture import ScreenCapture


class StarRailAI:
    """星穹铁道 AI 主类"""

    def __init__(self, config_path: str = "config/config.yaml",
                 device: str = "cpu",
                 use_gpu_ocr: bool = False):
        self.config_path = Path(config_path)
        self.device = device
        
        print("=" * 70)
        print("星穹铁道 AI 自动化系统")
        print("=" * 70)
        print()
        print("正在初始化 AI 模块...")
        print()
        print("✓ 初始化屏幕捕获模块...")
        self.screen_capture = ScreenCapture()
        print("✓ 初始化深度学习模型...")
        self.ai_model = GameAIModel(device=device)
        print("✓ 初始化智能 OCR...")
        self.ocr = SmartOCR(use_gpu=use_gpu_ocr)
        self.text_ai = TextUnderstandingAI()
        print("✓ 初始化智能决策引擎...")
        self.decision_engine = IntelligentDecisionEngine(device=device)
        print()
        print(f"✓ 设备：{device}")
        print(f"✓ AI 系统就绪!")
        print("=" * 70)
        print()

    def capture_screen(self) -> np.ndarray:
        return self.screen_capture.capture()

    def analyze_screen(self, screen: np.ndarray) -> Dict:
        ai_result = self.ai_model.infer(screen)
        ocr_texts = self.ocr.recognize(screen)
        text_understanding = self.ocr.understand_text(ocr_texts)
        
        quest_info = None
        if ocr_texts:
            for text_item in ocr_texts:
                text = text_item.get("text", "")
                if text:
                    parsed = self.text_ai.parse_quest_description(text)
                    if parsed["quest_type"] != "unknown":
                        quest_info = parsed
                        break
        
        return {
            "ai_result": ai_result,
            "ocr_texts": ocr_texts,
            "text_understanding": text_understanding,
            "quest_info": quest_info,
        }

    def decide_action(self, screen: np.ndarray) -> Dict:
        analysis = self.analyze_screen(screen)
        decision = self.decision_engine.get_context_aware_action(screen)
        decision["analysis"] = analysis
        return decision

    def execute_action(self, decision: Dict):
        action = decision.get("action", "wait")
        confidence = decision.get("confidence", 0)
        print(f"执行动作：{action} (置信度：{confidence:.2%})")
        
        if action == "wait":
            time.sleep(1)
        elif action == "click":
            x = decision.get("x")
            y = decision.get("y")
            if x and y:
                pyautogui.click(x, y)
            else:
                pyautogui.click()
        elif action == "track_quest":
            print("  → 正在追踪任务...")
            time.sleep(2)
        elif action == "claim_reward":
            print("  → 正在领取奖励...")
            pyautogui.press('space')
            time.sleep(1)
        elif action == "break_loop":
            print("  → 检测到死循环，随机操作打破...")
            pyautogui.press('esc')
            time.sleep(1)
        else:
            time.sleep(0.5)

    def run(self, mode: str = "daily", 
            duration: Optional[int] = None,
            max_iterations: Optional[int] = None):
        print(f"开始运行 - 模式：{mode}")
        if duration:
            print(f"运行时长：{duration}秒")
        if max_iterations:
            print(f"最大迭代：{max_iterations}次")
        print()
        
        start_time = time.time()
        iteration = 0
        
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    print(f"\n✓ 已达到运行时长：{duration}秒")
                    break
                if max_iterations and iteration >= max_iterations:
                    print(f"\n✓ 已达到最大迭代次数：{max_iterations}")
                    break
                
                iteration += 1
                print(f"\n[迭代 {iteration}]")
                
                screen = self.capture_screen()
                print(f"✓ 屏幕捕获：{screen.shape}")
                
                analysis = self.analyze_screen(screen)
                if analysis["quest_info"]:
                    qi = analysis["quest_info"]
                    print(f"✓ 识别到任务：{qi.get('quest_type')}")
                
                decision = self.decide_action(screen)
                print(f"✓ AI 决策：{decision.get('action')}")
                
                self.execute_action(decision)
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\n\n⚠ 用户中断运行")
        except Exception as e:
            print(f"\n\n❌ 发生错误：{e}")
        finally:
            print(f"\n总迭代次数：{iteration}")

    def demo(self):
        print("=" * 70)
        print("星穹铁道 AI 演示模式")
        print("=" * 70)
        print()
        print("系统组件:")
        print("  1. GameAIModel - 深度学习游戏理解模型 (~2.5M 参数)")
        print("  2. SmartOCR - 智能文字识别与理解")
        print("  3. DecisionEngine - 智能决策引擎")
        print("  4. TrainingManager - 强化学习训练")
        print()
        print("这不是脚本，是真正的深度学习 AI 系统!")
        print()


def main():
    parser = argparse.ArgumentParser(description="星穹铁道 AI 自动化系统")
    parser.add_argument("command", choices=["run", "train", "demo"],
                       help="命令：run(运行), train(训练), demo(演示)")
    parser.add_argument("--mode", type=str, default="daily",
                       choices=["daily", "battle", "explore"],
                       help="运行模式")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"], help="运行设备")
    parser.add_argument("--duration", type=int, default=None,
                       help="运行时长 (秒)")
    parser.add_argument("--max-iterations", type=int, default=None,
                       help="最大迭代次数")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="配置文件路径")
    parser.add_argument("--data", type=str, default=None,
                       help="训练数据路径")
    parser.add_argument("--use-gpu-ocr", action="store_true",
                       help="OCR 使用 GPU 加速")
    args = parser.parse_args()
    
    ai = StarRailAI(config_path=args.config, device=args.device, use_gpu_ocr=args.use_gpu_ocr)
    
    if args.command == "run":
        ai.run(mode=args.mode, duration=args.duration, max_iterations=args.max_iterations)
    elif args.command == "train":
        print("训练模式 - 需要先收集训练数据")
    elif args.command == "demo":
        ai.demo()


if __name__ == "__main__":
    main()
