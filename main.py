import pygame
import sys
from board import Board
from ai_player import AIPlayer

# 初始化 Pygame
pygame.init()

# 常量定义
WINDOW_SIZE = 800
BOARD_SIZE = 19
CELL_SIZE = WINDOW_SIZE // (BOARD_SIZE + 1)
STONE_SIZE = int(CELL_SIZE * 0.45)
GRID_COLOR = (0, 0, 0)
BOARD_COLOR = (219, 179, 119)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# 创建窗口
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("围棋")

def draw_board(screen, board):
    # 绘制棋盘背景
    screen.fill(BOARD_COLOR)
    
    # 绘制网格线
    for i in range(BOARD_SIZE):
        # 横线
        pygame.draw.line(screen, GRID_COLOR,
                        (CELL_SIZE, (i + 1) * CELL_SIZE),
                        (WINDOW_SIZE - CELL_SIZE, (i + 1) * CELL_SIZE))
        # 竖线
        pygame.draw.line(screen, GRID_COLOR,
                        ((i + 1) * CELL_SIZE, CELL_SIZE),
                        ((i + 1) * CELL_SIZE, WINDOW_SIZE - CELL_SIZE))
    
    # 绘制星位
    star_points = [(3, 3), (3, 9), (3, 15),
                   (9, 3), (9, 9), (9, 15),
                   (15, 3), (15, 9), (15, 15)]
    for x, y in star_points:
        pygame.draw.circle(screen, GRID_COLOR,
                         ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), 5)
    
    # 绘制棋子
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board.board[i][j] != 0:
                color = BLACK if board.board[i][j] == 1 else WHITE
                pos = ((i + 1) * CELL_SIZE, (j + 1) * CELL_SIZE)
                pygame.draw.circle(screen, color, pos, STONE_SIZE)

def get_board_pos(mouse_pos):
    x, y = mouse_pos
    board_x = round((x - CELL_SIZE) / CELL_SIZE) - 1
    board_y = round((y - CELL_SIZE) / CELL_SIZE) - 1
    if 0 <= board_x < BOARD_SIZE and 0 <= board_y < BOARD_SIZE:
        return board_x, board_y
    return None

def main():
    board = Board()
    ai = AIPlayer(board)
    clock = pygame.time.Clock()
    game_over = False

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    game_over = True
                elif event.key == pygame.K_r:
                    board = Board()
                    ai = AIPlayer(board)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # 人类玩家回合
                if board.current_player == 1:  # 黑子
                    pos = get_board_pos(event.pos)
                    if pos:
                        x, y = pos
                        if board.make_move(x, y):
                            # AI 玩家回合
                            pygame.display.flip()
                            pygame.time.wait(500)  # 等待一下，让玩家看清楚
                            ai_move = ai.make_move()
                            if ai_move:
                                board.make_move(*ai_move)

        draw_board(screen, board)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 