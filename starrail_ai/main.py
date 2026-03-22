#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
星穹铁道 AI 自动化系统 - 完整版
支持后台运行、自动启动游戏、不占用鼠标键盘
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict
import numpy as np

from ai_models.game_ai_model import GameAIModel
from ai_models.smart_ocr import SmartOCR, TextUnderstandingAI
from ai_models.decision_engine import IntelligentDecisionEngine
from core.screen_capture import ScreenCapture, GameWindow


class StarRailAI:
    """星穹铁道 AI 主类"""

    def __init__(self, device: str = "cpu", auto_start: bool = True):
        self.device = device
        
        print("=" * 70)
        print("星穹铁道 AI 自动化系统 v2.0")
        print("=" * 70)
        print()
        
        # 1. 查找/启动游戏
        print("正在查找星穹铁道窗口...")
        self.game_window = GameWindow()
        
        if not self.game_window.find_window():
            if auto_start:
                print("未找到游戏窗口，尝试自动启动...")
                self.game_window.start_game()
                print("等待游戏启动 (约 30-60 秒)...")
                for i in range(60):
                    time.sleep(1)
                    if self.game_window.find_window():
                        print(f"✓ 游戏已启动！窗口：{self.game_window.title}")
                        break
                    print(f"  等待中... ({i+1}/60)")
            else:
                print("❌ 未找到游戏窗口，请手动启动游戏后重试")
                sys.exit(1)
        
        if not self.game_window.find_window():
            print("❌ 游戏启动失败，请检查游戏路径配置")
            sys.exit(1)
        
        print()
        print(f"✓ 游戏窗口：{self.game_window.title}")
        print(f"✓ 窗口位置：{self.game_window.rect}")
        print()
        
        # 2. 初始化 AI 模块
        print("初始化 AI 模块...")
        print("✓ 屏幕捕获模块 (后台截图)")
        self.screen_capture = ScreenCapture(window=self.game_window)
        
        print("✓ 深度学习模型")
        self.ai_model = GameAIModel(device=device)
        
        print("✓ 智能 OCR")
        self.ocr = SmartOCR()
        self.text_ai = TextUnderstandingAI()
        
        print("✓ 决策引擎")
        self.decision_engine = IntelligentDecisionEngine()
        
        print()
        print(f"✓ AI 系统就绪！设备：{device}")
        print("=" * 70)
        print()

    def capture_screen(self) -> np.ndarray:
        """后台捕获游戏画面"""
        return self.screen_capture.capture()

    def analyze_screen(self, screen: np.ndarray) -> Dict:
        """分析屏幕"""
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
        """决策"""
        analysis = self.analyze_screen(screen)
        decision = self.decision_engine.get_context_aware_action(screen)
        decision["analysis"] = analysis
        return decision

    def execute_action(self, decision: Dict):
        """执行动作（后台）"""
        action = decision.get("action", "wait")
        confidence = decision.get("confidence", 0)
        print(f"执行：{action} (置信度：{confidence:.2%})")
        
        if action == "wait":
            time.sleep(1)
        elif action == "click":
            x = decision.get("x", self.game_window.width // 2)
            y = decision.get("y", self.game_window.height // 2)
            self.game_window.click(x, y)
        elif action == "track_quest":
            print("  → 追踪任务...")
            # 点击任务追踪按钮 (示例位置)
            self.game_window.click(100, 100)
            time.sleep(2)
        elif action == "claim_reward":
            print("  → 领取奖励...")
            self.game_window.press_key('space')
            time.sleep(1)
        elif action == "break_loop":
            print("  → 打破死循环...")
            self.game_window.press_key('esc')
            time.sleep(1)
        else:
            time.sleep(0.5)

    def run(self, mode: str = "daily", duration: Optional[int] = None,
            max_iterations: Optional[int] = None):
        print(f"开始运行 - 模式：{mode}")
        if duration: print(f"时长：{duration}秒")
        if max_iterations: print(f"最大迭代：{max_iterations}次")
        print()
        
        start_time = time.time()
        iteration = 0
        
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    print(f"\n✓ 运行完成：{duration}秒")
                    break
                if max_iterations and iteration >= max_iterations:
                    print(f"\n✓ 达到最大迭代：{max_iterations}")
                    break
                
                iteration += 1
                print(f"\n[{iteration}]")
                
                # 检查窗口是否还存在
                if not self.game_window.find_window():
                    print("❌ 游戏窗口丢失，停止运行")
                    break
                
                # 捕获
                screen = self.capture_screen()
                print(f"✓ 截图：{screen.shape}")
                
                # 分析
                analysis = self.analyze_screen(screen)
                if analysis["quest_info"]:
                    qi = analysis["quest_info"]
                    print(f"✓ 任务：{qi.get('quest_type')}")
                
                # 决策
                decision = self.decide_action(screen)
                print(f"✓ 决策：{decision.get('action')}")
                
                # 执行
                self.execute_action(decision)
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\n\n⚠ 用户中断")
        except Exception as e:
            print(f"\n\n❌ 错误：{e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"\n总迭代：{iteration}")

    def demo(self):
        print("=" * 70)
        print("演示模式")
        print("=" * 70)
        print()
        print("系统组件:")
        print("  • GameAIModel - 深度学习模型 (~2.5M 参数)")
        print("  • SmartOCR - 智能 OCR")
        print("  • DecisionEngine - 决策引擎")
        print("  • ScreenCapture - 后台截图")
        print()
        print("功能:")
        print("  ✓ 后台运行，不占用鼠标键盘")
        print("  ✓ 自动查找/启动游戏")
        print("  ✓ 智能识别任务")
        print("  ✓ 自主决策行动")
        print()


def main():
    parser = argparse.ArgumentParser(description="星穹铁道 AI 自动化系统")
    parser.add_argument("command", choices=["run", "train", "demo"])
    parser.add_argument("--mode", type=str, default="daily")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--no-auto-start", action="store_true", help="不自动启动游戏")
    args = parser.parse_args()
    
    ai = StarRailAI(device=args.device, auto_start=not args.no_auto_start)
    
    if args.command == "run":
        ai.run(mode=args.mode, duration=args.duration, max_iterations=args.max_iterations)
    elif args.command == "demo":
        ai.demo()
    elif args.command == "train":
        print("训练模式 - 待实现")


if __name__ == "__main__":
    main()
