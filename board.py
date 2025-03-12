import numpy as np

class Board:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: 空, 1: 黑, 2: 白
        self.current_player = 1  # 黑子先行
        self.history = []
        self.ko_point = None

    def is_valid_move(self, x, y):
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        if self.board[x][y] != 0:
            return False
        if self.ko_point == (x, y):
            return False
        
        # 临时落子检查
        self.board[x][y] = self.current_player
        has_liberty = self._check_liberties(x, y)
        captures_opponent = self._check_captures(x, y)
        self.board[x][y] = 0

        return has_liberty or captures_opponent

    def _check_liberties(self, x, y):
        """检查一个位置的气"""
        color = self.board[x][y]
        if color == 0:
            return True
        
        visited = set()
        return self._find_liberties(x, y, color, visited)

    def _find_liberties(self, x, y, color, visited):
        """递归检查气"""
        if (x, y) in visited:
            return False
        
        visited.add((x, y))
        
        # 检查四个方向
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if not (0 <= new_x < self.size and 0 <= new_y < self.size):
                continue
                
            if self.board[new_x][new_y] == 0:
                return True
            if self.board[new_x][new_y] == color:
                if self._find_liberties(new_x, new_y, color, visited):
                    return True
        
        return False

    def _check_captures(self, x, y):
        """检查是否有提子"""
        captures = []
        color = self.board[x][y]
        opponent = 3 - color  # 切换颜色 (1->2, 2->1)
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if not (0 <= new_x < self.size and 0 <= new_y < self.size):
                continue
            if self.board[new_x][new_y] == opponent:
                if not self._check_liberties(new_x, new_y):
                    captures.append((new_x, new_y))
        
        return len(captures) > 0

    def make_move(self, x, y):
        """尝试在指定位置落子"""
        if not self.is_valid_move(x, y):
            return False

        self.board[x][y] = self.current_player
        self.history.append((x, y))
        
        # 处理提子
        captures = []
        opponent = 3 - self.current_player
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if not (0 <= new_x < self.size and 0 <= new_y < self.size):
                continue
            if self.board[new_x][new_y] == opponent:
                if not self._check_liberties(new_x, new_y):
                    self._remove_group(new_x, new_y)
        
        self.current_player = opponent
        return True

    def _remove_group(self, x, y):
        """移除一个死棋组"""
        color = self.board[x][y]
        stack = [(x, y)]
        removed = set()
        
        while stack:
            curr_x, curr_y = stack.pop()
            if (curr_x, curr_y) in removed:
                continue
                
            removed.add((curr_x, curr_y))
            self.board[curr_x][curr_y] = 0
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = curr_x + dx, curr_y + dy
                if not (0 <= new_x < self.size and 0 <= new_y < self.size):
                    continue
                if self.board[new_x][new_y] == color:
                    stack.append((new_x, new_y))

    def get_valid_moves(self):
        """返回所有合法落子点"""
        valid_moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_valid_move(x, y):
                    valid_moves.append((x, y))
        return valid_moves 