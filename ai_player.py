import random
import numpy as np
from model import load_pretrained_model
from mcts import MCTS
import torch

class AIPlayer:
    def __init__(self, board):
        self.board = board
        # 尝试加载自学习模型，如果失败则使用预训练模型
        try:
            checkpoint = torch.load('go_model_final.pth')
            self.model = load_pretrained_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("已加载自学习模型")
        except:
            self.model = load_pretrained_model()
            print("使用预训练模型")
            
        self.mcts = MCTS(self.model, num_simulations=400)
        
    def make_move(self):
        """AI 选择下一步落子位置"""
        # 使用 MCTS 搜索最佳移动
        move = self.mcts.search(self.board)
        
        if move is None:
            # 如果 MCTS 无法找到有效移动，使用备用策略
            return self._fallback_strategy()
            
        return move
        
    def _fallback_strategy(self):
        """备用策略：使用简单的启发式方法"""
        valid_moves = self.board.get_valid_moves()
        if not valid_moves:
            return None

        center = self.board.size // 2
        scored_moves = []
        
        for x, y in valid_moves:
            score = 0
            
            # 距离中心的距离作为基础分数
            distance_to_center = abs(x - center) + abs(y - center)
            score -= distance_to_center
            
            # 检查周围是否有己方棋子
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.board.size and 0 <= new_y < self.board.size:
                    if self.board.board[new_x][new_y] == self.board.current_player:
                        score += 2
            
            # 如果这步棋可以提子
            if self._can_capture(x, y):
                score += 10
            
            scored_moves.append((score, x, y))
        
        scored_moves.sort(reverse=True)
        _, x, y = scored_moves[0]
        return x, y

    def _can_capture(self, x, y):
        """检查是否可以提子"""
        opponent = 3 - self.board.current_player
        self.board.board[x][y] = self.board.current_player
        can_capture = False
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.board.size and 0 <= new_y < self.board.size:
                if self.board.board[new_x][new_y] == opponent:
                    if not self.board._check_liberties(new_x, new_y):
                        can_capture = True
                        break
        
        self.board.board[x][y] = 0
        return can_capture 