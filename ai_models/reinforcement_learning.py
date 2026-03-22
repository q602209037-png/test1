"""
强化学习训练模块
让 AI 通过玩游戏自己学习策略
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Optional
from collections import deque
import random
import json
from pathlib import Path


class ExperienceReplayBuffer:
    """
    经验回放缓冲区
    存储 (state, action, reward, next_state, done) 用于训练
    """
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, 
             reward: float, next_state: np.ndarray, done: bool):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> List:
        """随机采样一批经验"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 强化学习代理
    用于学习游戏策略
    """
    
    def __init__(self, state_dim: int = 512, num_actions: int = 50,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 clip_epsilon: float = 0.2):
        """
        初始化 PPO 代理
        
        Args:
            state_dim: 状态维度
            num_actions: 动作数量
            learning_rate: 学习率
            gamma: 折扣因子
            clip_epsilon: PPO 裁剪参数
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor-Critic 网络
        self.actor = self._build_actor().to(self.device)
        self.critic = self._build_critic().to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # 经验回放
        self.memory = ExperienceReplayBuffer()
        
    def _build_actor(self) -> nn.Module:
        """构建 Actor 网络"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
            nn.Softmax(dim=-1)
        )
    
    def _build_critic(self) -> nn.Module:
        """构建 Critic 网络"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def select_action(self, state: np.ndarray, 
                      training: bool = False) -> Tuple[int, float]:
        """
        选择动作
        
        Args:
            state: 状态向量
            training: 是否训练模式 (探索)
            
        Returns:
            (动作，动作概率)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)[0]
            
            if training:
                # 训练时采样
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                prob = action_probs[action]
            else:
                # 测试时选择最优
                action = torch.argmax(action_probs)
                prob = action_probs[action]
        
        return action.item(), prob.item()
    
    def compute_returns(self, rewards: List[float], 
                        values: List[float], dones: List[bool]) -> torch.Tensor:
        """计算回报"""
        returns = []
        g_t = 0
        
        for reward, value, done in zip(reversed(rewards), 
                                        reversed(values), 
                                        reversed(dones)):
            if done:
                g_t = 0
            g_t = reward + self.gamma * g_t
            returns.insert(0, g_t)
        
        return torch.FloatTensor(returns).to(self.device)
    
    def update(self, batch_size: int = 64, epochs: int = 10) -> Dict[str, float]:
        """
        更新策略
        
        Args:
            batch_size: 批次大小
            epochs: 更新轮数
            
        Returns:
            损失信息
        """
        if len(self.memory) < batch_size:
            return {"actor_loss": 0, "critic_loss": 0}
        
        # 采样
        transitions = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前策略的概率
        action_probs = self.actor(states)
        old_action_probs = action_probs.detach()
        
        # 计算优势函数 (简化版)
        values = self.critic(states)
        next_values = self.critic(next_states)
        
        td_target = rewards + self.gamma * next_values * (1 - dones)
        td_error = td_target - values
        
        advantages = td_error.detach()
        
        # PPO 更新
        actor_loss_total = 0
        critic_loss_total = 0
        
        for _ in range(epochs):
            # 计算新策略的概率
            new_action_probs = self.actor(states)
            
            # 比率
            ratio = new_action_probs.gather(1, actions.unsqueeze(1)) / \
                    (old_action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
            
            # 裁剪的目标函数
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                               1 + self.clip_epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic 损失
            critic_loss = nn.MSELoss()(
                self.critic(states), td_target
            )
            
            # 更新
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            actor_loss_total += actor_loss.item()
            critic_loss_total += critic_loss.item()
        
        return {
            "actor_loss": actor_loss_total / epochs,
            "critic_loss": critic_loss_total / epochs
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict()
        }, path)
        print(f"PPO 模型已保存：{path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        print(f"PPO 模型已加载：{path}")


class ImitationLearning:
    """
    模仿学习
    通过记录人类玩家的操作来学习
    """
    
    def __init__(self, state_dim: int = 512, num_actions: int = 50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 策略网络
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        
        # 演示数据
        self.demonstrations = []
        
    def record_demonstration(self, state: np.ndarray, action: int):
        """记录一次演示"""
        self.demonstrations.append((state, action))
        
    def train(self, epochs: int = 100, batch_size: int = 64) -> float:
        """
        从演示数据中学习
        
        Returns:
            最终损失
        """
        if len(self.demonstrations) < batch_size:
            print(f"演示数据不足：{len(self.demonstrations)} < {batch_size}")
            return float('inf')
        
        states = torch.FloatTensor(np.array([d[0] for d in self.demonstrations]))
        actions = torch.LongTensor([d[1] for d in self.demonstrations])
        
        dataset = torch.utils.data.TensorDataset(states, actions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0
        num_batches = 0
        
        for epoch in range(epochs):
            for batch_states, batch_actions in loader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                
                # 前向传播
                logits = self.policy(batch_states)
                
                # 计算损失
                loss = self.criterion(logits, batch_actions)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"模仿学习完成，平均损失：{avg_loss:.4f}")
        return avg_loss
    
    def predict_action(self, state: np.ndarray) -> int:
        """预测动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.policy(state_tensor)
            action = torch.argmax(logits, dim=-1).item()
        
        return action
    
    def save(self, path: str):
        """保存模型"""
        torch.save(self.policy.state_dict(), path)
        print(f"模仿学习模型已保存：{path}")
    
    def load(self, path: str):
        """加载模型"""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        print(f"模仿学习模型已加载：{path}")


class TrainingManager:
    """
    训练管理器
    协调各种训练方法
    """
    
    def __init__(self, checkpoint_dir: str = "ai_models/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化各种训练代理
        self.ppo_agent = PPOAgent()
        self.imitation_learner = ImitationLearning()
        
        # 训练日志
        self.training_log = []
        
    def start_imitation_learning(self, demonstrations: List[Tuple]):
        """
        开始模仿学习
        
        Args:
            demonstrations: 演示数据列表 [(state, action), ...]
        """
        print(f"开始模仿学习，演示数据量：{len(demonstrations)}")
        
        for state, action in demonstrations:
            self.imitation_learner.record_demonstration(state, action)
        
        loss = self.imitation_learner.train(epochs=50, batch_size=64)
        
        # 保存模型
        model_path = self.checkpoint_dir / "imitation_learning.pt"
        self.imitation_learner.save(str(model_path))
        
        return loss
    
    def start_ppo_training(self, num_episodes: int = 1000):
        """
        开始 PPO 强化学习
        
        Args:
            num_episodes: 训练回合数
        """
        print(f"开始 PPO 训练，回合数：{num_episodes}")
        
        for episode in range(num_episodes):
            # 这里需要与游戏环境交互
            # 实际实现时需要连接游戏
            pass
        
        # 保存模型
        model_path = self.checkpoint_dir / "ppo_agent.pt"
        self.ppo_agent.save(str(model_path))
    
    def export_model(self, output_path: str):
        """导出最终模型"""
        # 整合所有训练好的模型
        checkpoint = {
            "imitation_learning": self.imitation_learner.policy.state_dict(),
            "ppo_actor": self.ppo_agent.actor.state_dict(),
            "ppo_critic": self.ppo_agent.critic.state_dict(),
            "training_log": self.training_log
        }
        
        torch.save(checkpoint, output_path)
        print(f"模型已导出：{output_path}")


if __name__ == "__main__":
    print("=== 强化学习训练模块 ===\n")
    
    # 创建训练管理器
    trainer = TrainingManager()
    
    print("可用的训练方法:")
    print("1. 模仿学习 (Imitation Learning) - 通过记录人类操作学习")
    print("2. PPO 强化学习 - 通过试错自我学习")
    print("\n训练流程:")
    print("1. 先收集人类玩家的演示数据")
    print("2. 用模仿学习预训练模型")
    print("3. 用 PPO 强化学习进一步优化")
    print("\n这是一个真正的 AI 学习系统，不是硬编码脚本!")
    
    # 测试 PPO 代理
    print("\n=== 测试 PPO 代理 ===")
    ppo = PPOAgent(state_dim=512, num_actions=50)
    
    # 模拟一个状态
    dummy_state = np.random.randn(512).astype(np.float32)
    action, prob = ppo.select_action(dummy_state, training=False)
    print(f"状态维度：{dummy_state.shape}")
    print(f"预测动作：{action}, 概率：{prob:.4f}")
