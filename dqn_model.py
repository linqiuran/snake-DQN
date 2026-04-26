import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """深度Q网络，使用CNN处理网格状态"""

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # 计算卷积层输出的尺寸
        conv_output_size = input_shape[0] * input_shape[1] * 64

        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        """前向传播
        输入: x - 状态张量，形状为 [batch_size, height, width, channels]
        输出: 每个动作的Q值
        """
        # 调整维度顺序为 [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 展平
        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)


class QNetwork(nn.Module):
    def __init__(self, input_dim=12, n_actions=4, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, n_actions)

    def forward(self, x):
        # x shape: (batch, 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)  # Q-values for each action