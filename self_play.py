import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from tqdm import tqdm
import random
import os
from model import GoNet, prepare_input
from board import Board
from mcts import MCTS, MCTSNode
import copy

class SelfPlayAI:
    def __init__(self, board_size=19, num_simulations=400):
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.model = GoNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.mcts = MCTS(self.model, num_simulations)
        self.memory = deque(maxlen=10000)  # 经验回放缓冲区
        
    def self_play_game(self, temperature=1.0):
        """进行一局自我对弈"""
        board = Board()
        game_states = []
        
        while True:
            # 准备当前状态
            state = prepare_input(board.board)
            
            # 使用 MCTS 选择动作
            move = self.mcts.search(board)
            if move is None:
                break
                
            # 获取所有子节点的访问次数
            root = MCTSNode(copy.deepcopy(board))
            root = self.mcts.search_tree(root)
            visits = [(m, child.visit_count) for m, child in root.children.items()]
            
            if not visits:
                break
                
            total_visits = sum(visit for _, visit in visits)
            if total_visits == 0:
                break
                
            # 计算移动概率
            policy = np.zeros(self.board_size * self.board_size + 1)
            for move, visit_count in visits:
                x, y = move
                policy[x * self.board_size + y] = visit_count / total_visits
            
            # 保存状态
            game_states.append((
                state,
                policy,
                board.current_player
            ))
            
            # 根据温度参数选择移动
            if temperature == 0:
                # 选择访问次数最多的移动
                move = max(visits, key=lambda x: x[1])[0]
            else:
                # 根据访问次数的概率分布随机选择
                visit_counts = np.array([visit for _, visit in visits])
                probs = visit_counts ** (1/temperature)
                probs = probs / probs.sum()
                move_idx = np.random.choice(len(visits), p=probs)
                move = visits[move_idx][0]
            
            # 执行移动
            if not board.make_move(*move):
                break
            
            # 检查游戏是否结束
            if len(board.get_valid_moves()) == 0:
                break
        
        # 计算游戏结果
        black_score = np.sum(board.board == 1)
        white_score = np.sum(board.board == 2)
        winner = 1 if black_score > white_score else 2
        
        # 为每个状态添加价值标签
        for state, policy, player in game_states:
            value = 1 if player == winner else -1
            self.memory.append((state, policy, value))
    
    def train_step(self, batch_size=32):
        """训练一个批次"""
        if len(self.memory) < batch_size:
            return 0
            
        # 随机采样批次
        batch = random.sample(self.memory, batch_size)
        states, policies, values = zip(*batch)
        
        # 准备训练数据
        states = torch.cat(states)
        policies = torch.FloatTensor(np.array(policies))
        values = torch.FloatTensor(np.array(values))
        
        # 前向传播
        policy_out, value_out = self.model(states)
        
        # 计算损失
        policy_loss = -torch.mean(torch.sum(policies * torch.log(policy_out + 1e-8), dim=1))
        value_loss = torch.mean((value_out.squeeze() - values) ** 2)
        loss = policy_loss + value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_iterations=100, games_per_iteration=10, 
             batch_size=32, batches_per_iteration=10):
        """训练循环"""
        for iteration in range(num_iterations):
            print(f"\n开始迭代 {iteration + 1}/{num_iterations}")
            
            # 自我对弈收集数据
            print("进行自我对弈...")
            for _ in tqdm(range(games_per_iteration)):
                self.self_play_game(temperature=4.0)
            
            # 训练模型
            print("训练模型...")
            total_loss = 0
            for _ in tqdm(range(batches_per_iteration)):
                total_loss += self.train_step(batch_size)
            avg_loss = total_loss / batches_per_iteration
            print(f"平均损失: {avg_loss:.4f}")
            
            # 每10次迭代保存一次模型
            if (iteration + 1) % 10 == 0:
                self.save_model(f"go_model_iter_{iteration+1}.pth")
                
    def save_model(self, filename):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)
        
    def load_model(self, filename):
        """加载模型"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"成功加载模型：{filename}")
        else:
            print(f"未找到模型文件：{filename}")

def main():
    # 创建自学习AI
    ai = SelfPlayAI(num_simulations=2)  # 减少模拟次数以加快训练速度
    
    # 训练参数
    num_iterations = 10        # 总迭代次数
    games_per_iteration = 2    # 每次迭代的对弈局数
    batch_size = 32           # 批次大小
    batches_per_iteration = 5  # 每次迭代的训练批次数
    
    print("开始自我学习训练...")
    ai.train(
        num_iterations=num_iterations,
        games_per_iteration=games_per_iteration,
        batch_size=batch_size,
        batches_per_iteration=batches_per_iteration
    )
    
    # 保存最终模型
    ai.save_model("go_model_final.pth")
    print("训练完成！")

if __name__ == "__main__":
    main()