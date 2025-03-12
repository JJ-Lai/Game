import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GoNet(nn.Module):
    def __init__(self):
        super(GoNet, self).__init__()
        # 输入通道：1（棋盘状态）
        # 3个残差块，每个块包含2个卷积层
        self.conv1 = nn.Conv2d(1, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResBlock(256) for _ in range(3)
        ])
        
        # 策略头（输出移动概率）
        self.policy_conv = nn.Conv2d(256, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 19 * 19, 19 * 19 + 1)  # +1 为pass移动
        
        # 价值头（输出局面评估）
        self.value_conv = nn.Conv2d(256, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(19 * 19, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        for block in self.res_blocks:
            x = block(x)
        
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * 19 * 19)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)
        
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 19 * 19)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

def prepare_input(board):
    """将棋盘状态转换为神经网络输入格式"""
    state = np.zeros((1, 1, 19, 19), dtype=np.float32)
    for i in range(19):
        for j in range(19):
            if board[i][j] == 1:  # 黑子
                state[0, 0, i, j] = 1
            elif board[i][j] == 2:  # 白子
                state[0, 0, i, j] = -1
    return torch.FloatTensor(state)

def load_pretrained_model():
    """加载预训练模型"""
    model = GoNet()
    try:
        model.load_state_dict(torch.load('go_model.pth'))
        model.eval()
    except FileNotFoundError:
        print("未找到预训练模型，使用初始化的模型")
    return model 