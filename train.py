import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sgf
import requests
import os
from tqdm import tqdm
from model import GoNet, prepare_input
from board import Board

def download_pro_games(num_games=100):
    """下载职业棋手的对局记录"""
    if not os.path.exists('games'):
        os.makedirs('games')
    
    # 从 GoGoD 数据库下载示例游戏
    sample_urls = [
        "https://homepages.cwi.nl/~aeb/go/games/games/Honinbo/Hon-1.sgf",
        "https://homepages.cwi.nl/~aeb/go/games/games/Kisei/Ki-1.sgf"
    ]
    
    for i, url in enumerate(sample_urls):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(f'games/game_{i+1}.sgf', 'w', encoding='utf-8') as f:
                    f.write(response.text)
        except Exception as e:
            print(f"下载失败: {url}, 错误: {e}")

def parse_sgf_file(file_path):
    """解析SGF文件，返回移动序列和胜者"""
    with open(file_path, 'r', encoding='utf-8') as f:
        sgf_content = f.read()
    
    collection = sgf.parse(sgf_content)
    game = collection.children[0]  # 获取第一个对局
    
    moves = []
    for node in game.nodes[1:]:  # 跳过根节点
        if 'B' in node.properties:
            moves.append(('B', node.properties['B'][0]))
        elif 'W' in node.properties:
            moves.append(('W', node.properties['W'][0]))
    
    # 获取胜者信息
    result = game.nodes[0].properties.get('RE', [''])[0]
    winner = 'B' if 'B+' in result else 'W'
    
    return moves, winner

def create_training_data(moves, winner):
    """从对局创建训练数据"""
    board = Board()
    training_data = []
    
    for i, (color, move) in enumerate(moves):
        if not move or len(move) != 2:
            continue
            
        # 转换坐标
        x = ord(move[0].lower()) - ord('a')
        y = ord(move[1].lower()) - ord('a')
        
        if not (0 <= x < 19 and 0 <= y < 19):
            continue
        
        # 准备输入数据
        current_state = prepare_input(board.board)
        
        # 准备标签
        policy = np.zeros(19 * 19 + 1)  # +1 为pass移动
        policy[x * 19 + y] = 1
        
        # 价值标签：如果当前玩家是赢家，则为1；否则为-1
        current_player = 1 if color == 'B' else 2
        value = 1 if (winner == 'B' and current_player == 1) or \
                    (winner == 'W' and current_player == 2) else -1
        
        training_data.append((current_state, policy, value))
        
        # 执行移动
        board.make_move(x, y)
    
    return training_data

def train_model(model, training_data, num_epochs=10, batch_size=32):
    """训练模型"""
    if not training_data:
        print("没有训练数据！请确保正确下载并解析了SGF文件。")
        return

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        np.random.shuffle(training_data)
        
        for i in tqdm(range(0, len(training_data), batch_size)):
            batch = training_data[i:i + batch_size]
            states, policies, values = zip(*batch)
            
            states = torch.cat(states)
            policies = torch.FloatTensor(np.array(policies))
            values = torch.FloatTensor(np.array(values))
            
            optimizer.zero_grad()
            
            # 前向传播
            policy_out, value_out = model(states)
            
            # 计算损失
            policy_loss = policy_criterion(policy_out, policies)
            value_loss = value_criterion(value_out.squeeze(), values)
            loss = policy_loss + value_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / (len(training_data) / batch_size)}")
    
    # 保存模型
    torch.save(model.state_dict(), 'go_model.pth')

def main():
    print("开始训练过程...")
    
    # 下载职业棋手对局
    print("下载职业棋手对局...")
    download_pro_games()
    
    # 初始化模型
    print("初始化模型...")
    model = GoNet()
    
    # 收集训练数据
    print("收集训练数据...")
    training_data = []
    games_dir = 'games'
    
    if not os.path.exists(games_dir):
        print(f"错误：找不到 {games_dir} 目录")
        return
        
    sgf_files = [f for f in os.listdir(games_dir) if f.endswith('.sgf')]
    if not sgf_files:
        print(f"错误：在 {games_dir} 目录中没有找到SGF文件")
        return
        
    for file_name in sgf_files:
        try:
            moves, winner = parse_sgf_file(os.path.join(games_dir, file_name))
            training_data.extend(create_training_data(moves, winner))
            print(f"成功解析文件：{file_name}")
        except Exception as e:
            print(f"解析文件 {file_name} 时出错：{e}")
    
    # 训练模型
    print(f"开始训练模型，数据集大小：{len(training_data)}")
    train_model(model, training_data)
    print("训练完成！")

if __name__ == '__main__':
    main() 