"""
星穹铁道 AI 主程序入口
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from ai_models.game_ai_model import GameAIModel
from ai_models.smart_ocr import SmartOCR
from ai_models.decision_engine import IntelligentDecisionEngine
from ai_models.reinforcement_learning import TrainingManager
from core.screen_capture import ScreenCapture


class StarRailAI:
    """
    星穹铁道 AI 主类
    整合所有模块，提供统一的接口
    """
    
    def __init__(self, config_path: str = "config/config.yaml",
                 device: str = "cpu"):
        """
        初始化 AI 系统
        
        Args:
            config_path: 配置文件路径
            device: 运行设备 (cpu 或 cuda)
        """
        self.config_path = Path(config_path)
        self.device = device
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化模块
        print("初始化 AI 模块...")
        self.screen_capture = ScreenCapture()
        self.decision_engine = IntelligentDecisionEngine(
            device=device,
            model_checkpoint=self.config.get("ai_model", {}).get(
                "checkpoint_path"
            )
        )
        
        print(f"✓ 屏幕捕获模块已初始化")
        print(f"✓ 决策引擎已初始化 (设备：{device})")
        print(f"✓ AI 系统就绪\n")
        
    def _load_config(self) -> dict:
        """加载配置文件"""
        import yaml
        
        if not self.config_path.exists():
            print(f"警告：配置文件不存在 {self.config_path}")
            return {}
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def run(self, mode: str = "daily", duration: Optional[int] = None):
        """
        运行 AI 自动化
        
        Args:
            mode: 运行模式 (daily, battle, explore, custom)
            duration: 运行时长 (秒)，None 表示无限运行
        """
        print(f"开始运行 - 模式：{mode}")
        print(f"设备：{self.device}")
        print("-" * 50)
        
        start_time = time.time()
        iteration = 0
        
        try:
            while True:
                # 检查运行时长
                if duration and (time.time() - start_time) > duration:
                    print(f"\n已达到运行时长：{duration}秒")
                    break
                
                # 捕获屏幕
                screen = self.screen_capture.capture()
                
                # AI 决策
                decision = self.decision_engine.get_context_aware_action(screen)
                
                # 显示决策
                iteration += 1
                print(f"\n[迭代 {iteration}]")
                print(f"  界面类型：{decision.get('screen_type', 'unknown')}")
                print(f"  决策动作：{decision.get('action', 'unknown')}")
                print(f"  置信度：{decision.get('confidence', 0):.2%}")
                print(f"  原因：{decision.get('reason', 'N/A')}")
                
                # 执行动作 (这里需要实现实际的动作执行)
                # self._execute_action(decision)
                
                # 延迟
                delay = self.config.get("automation", {}).get("action_delay", 0.5)
                time.sleep(delay)
                
        except KeyboardInterrupt:
            print("\n\n用户中断运行")
        except Exception as e:
            print(f"\n发生错误：{e}")
            import traceback
            traceback.print_exc()
        finally:
            self._save_results()
    
    def _execute_action(self, decision: dict):
        """
        执行 AI 决策的动作
        
        Args:
            decision: 决策结果
        """
        action = decision.get("action")
        
        if action == "wait":
            time.sleep(1)
        elif action == "click":
            # 使用 pyautogui 点击
            import pyautogui
            pyautogui.click()
        elif action == "claim_all":
            # 领取所有奖励
            pass
        # ... 其他动作
        
    def _save_results(self):
        """保存运行结果"""
        # 导出决策日志
        self.decision_engine.export_decision_log("logs/decision_log.json")
        print("运行结果已保存")
    
    def train(self, mode: str = "imitation", 
              data_path: Optional[str] = None):
        """
        训练 AI 模型
        
        Args:
            mode: 训练模式 (imitation, ppo)
            data_path: 训练数据路径
        """
        print(f"开始训练 - 模式：{mode}")
        
        trainer = TrainingManager()
        
        if mode == "imitation":
            # 模仿学习
            if data_path:
                # 加载演示数据
                demonstrations = self._load_demonstrations(data_path)
                trainer.start_imitation_learning(demonstrations)
            else:
                print("请提供演示数据路径")
        
        elif mode == "ppo":
            # PPO 强化学习
            trainer.start_ppo_training(num_episodes=1000)
        
        print("训练完成")
    
    def _load_demonstrations(self, path: str) -> list:
        """加载演示数据"""
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        demonstrations = []
        for item in data:
            state = item.get("state")
            action = item.get("action")
            if state and action is not None:
                demonstrations.append((state, action))
        
        print(f"加载演示数据：{len(demonstrations)} 条")
        return demonstrations
    
    def demo(self):
        """运行演示模式"""
        print("=" * 50)
        print("星穹铁道 AI 演示模式")
        print("=" * 50)
        
        # 显示系统信息
        print("\n系统组件:")
        print("  1. GameAIModel - 深度学习游戏理解模型")
        print("  2. SmartOCR - 智能文字识别与理解")
        print("  3. DecisionEngine - 智能决策引擎")
        print("  4. TrainingManager - 强化学习训练")
        
        print("\nAI 能力:")
        print("  ✓ 视觉理解 - 识别游戏界面元素")
        print("  ✓ 文字识别 - OCR 识别任务文本")
        print("  ✓ 智能决策 - 基于状态做出决策")
        print("  ✓ 自主学习 - 从游戏中学习策略")
        
        print("\n与脚本的区别:")
        print("  ❌ 脚本：硬编码规则，无法适应变化")
        print("  ✅ AI：深度学习，能够泛化和适应")
        
        print("\n使用方法:")
        print("  1. 收集训练数据 (玩游戏时记录)")
        print("  2. 训练模型 (imitation learning + PPO)")
        print("  3. 运行自动化 (run 命令)")
        
        print("\n" + "=" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="星穹铁道 AI 自动化系统"
    )
    
    parser.add_argument(
        "command",
        choices=["run", "train", "demo"],
        help="命令：run(运行), train(训练), demo(演示)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="daily",
        help="运行/训练模式"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="运行设备"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="运行时长 (秒)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="训练数据路径"
    )
    
    args = parser.parse_args()
    
    # 创建 AI 系统
    ai = StarRailAI(config_path=args.config, device=args.device)
    
    # 执行命令
    if args.command == "run":
        ai.run(mode=args.mode, duration=args.duration)
    elif args.command == "train":
        ai.train(mode=args.mode, data_path=args.data)
    elif args.command == "demo":
        ai.demo()


if __name__ == "__main__":
    main()
