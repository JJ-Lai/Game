import math
import numpy as np
from copy import deepcopy
from model import prepare_input

class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self):
        """使用 PUCT 算法选择子节点"""
        c_puct = 1.0
        
        best_score = float('-inf')
        best_child = None
        best_action = None
        
        for action, child in self.children.items():
            # UCB 公式
            q_value = -child.value()  # 负号是因为我们从对手的角度看价值
            u_value = (c_puct * child.prior * 
                      math.sqrt(self.visit_count) / (1 + child.visit_count))
            
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
                best_action = action
                
        return best_action, best_child

    def expand(self, policy):
        """展开节点"""
        self.is_expanded = True
        valid_moves = self.board.get_valid_moves()
        policy_sum = 1e-8
        
        # 只为合法移动分配概率
        for move in valid_moves:
            x, y = move
            idx = x * 19 + y
            policy_sum += policy[idx]
            
        # 创建子节点
        for move in valid_moves:
            x, y = move
            idx = x * 19 + y
            prior = policy[idx] / policy_sum
            
            new_board = deepcopy(self.board)
            new_board.make_move(x, y)
            
            self.children[(x, y)] = MCTSNode(
                new_board, parent=self, move=move, prior=prior)

class MCTS:
    def __init__(self, model, num_simulations=400):
        self.model = model
        self.num_simulations = num_simulations

    def search(self, board):
        """搜索最佳移动"""
        root = self.search_tree(MCTSNode(deepcopy(board)))
        
        if not root.children:
            return None
            
        visits = [(move, child.visit_count) for move, child in root.children.items()]
        if not visits:
            return None
            
        # 选择访问次数最多的移动
        best_move = max(visits, key=lambda x: x[1])[0]
        return best_move

    def search_tree(self, root):
        """在给定根节点上执行 MCTS 搜索"""
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # 选择
            while node.is_expanded:
                action, node = node.select_child()
                if action is None:
                    break
                search_path.append(node)
            
            # 扩展和评估
            state = prepare_input(node.board.board)
            policy, value = self.model(state)
            policy = policy.detach().numpy()[0]
            value = value.item()
            
            # 如果游戏未结束，则展开节点
            if len(node.board.get_valid_moves()) > 0:
                node.expand(policy)
            
            # 反向传播
            self.backpropagate(search_path, value)
            
        return root

    def backpropagate(self, search_path, value):
        """反向传播更新节点统计信息"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value  # 在每一层交替价值符号 