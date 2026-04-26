import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from dqn_model import DQN
from dqn_model import QNetwork
from NoisyLinear import NoisyQNetwork,NoisyLinear


class DQNAgent:
    """DQN智能体"""

    def __init__(self, env, buffer_size=10000, batch_size=64, gamma=0.99,
                 learning_rate=1e-4, tau=0.005, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 超参数
        self.batch_size = batch_size
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 目标网络软更新系数
        self.epsilon = epsilon_start  # 探索率
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # cnn网络
        # state_shape = env._get_state().shape
        # self.policy_net = DQN(state_shape, env.n_actions).to(self.device)
        # self.target_net = DQN(state_shape, env.n_actions).to(self.device)
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.target_net.eval()  # 目标网络不训练

        # mlp网络
        state_dim = env._get_state_mlp().shape[0]
        # self.policy_net_mlp = QNetwork(input_dim=state_dim,n_actions=env.n_actions ).to(self.device)
        # self.target_net_mlp = QNetwork(input_dim=state_dim, n_actions=env.n_actions).to(self.device)
        # self.target_net_mlp.load_state_dict(self.policy_net_mlp.state_dict())
        # self.target_net_mlp.eval()

        #noisy
        self.policy_net_noisy = NoisyQNetwork(state_dim=state_dim, action_dim=env.n_actions).to(self.device)
        self.target_net_noisy = NoisyQNetwork(state_dim=state_dim, action_dim=env.n_actions).to(self.device)
        self.target_net_noisy.load_state_dict(self.policy_net_noisy.state_dict())
        self.avg_sigma = self.policy_net_noisy.noisy1.weight_sigma.abs().mean().item()

        # 优化器
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # self.optimizer = optim.Adam(self.policy_net_mlp.parameters(), lr=learning_rate)
        self.optimizer = optim.Adam(self.policy_net_noisy.parameters(), lr=learning_rate)

        # 经验回放
        from replay_buffer import ReplayBuffer
        self.memory = ReplayBuffer(buffer_size)

        # 训练统计
        self.episode_rewards = []
        self.episode_losses = []
        self.update_count = 0

    def select_action(self, state,eval_model=False):
        """根据epsilon-greedy策略选择动作"""

        #noisy
        if eval_model:
            #评估模式
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.policy_net_noisy(state_tensor)
                return q_values.argmax().item()

        else:
            self.policy_net_noisy.sample_noise()
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.policy_net_noisy(state_tensor)
                return q_values.max(1)[1].item()

        # if random.random() < self.epsilon:
        #     return random.randrange(self.env.n_actions)

        # with torch.no_grad():
        #     state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        #     # q_values = self.policy_net(state_tensor)
        #     q_values = self.policy_net_mlp(state_tensor)
        #     return q_values.argmax().item()

    def optimize_model(self):
        """优化模型"""
        if len(self.memory) < self.batch_size:
            return 0

        next_q_values = torch.zeros(self.batch_size, device=self.device)

        # 采样经验
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 转换为张量
        state_batch = torch.tensor(states, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # 计算当前Q值
        # current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        # current_q_values = self.policy_net_mlp(state_batch).gather(1, action_batch)
        current_q_values = self.policy_net_noisy(state_batch).gather(1, action_batch)

        # non_final_mask = torch.tensor(dones,dtype=torch.bool,device=self.device)
        non_final_mask = ~done_batch.bool()
        non_final_next_states = next_state_batch[non_final_mask]

        if non_final_next_states.size(0) > 0:
            with torch.no_grad():
                # next_actions = self.policy_net_mlp(non_final_next_states).max(1)[1].unsqueeze(1)
                # next_q_values[non_final_mask] = self.target_net_mlp(non_final_next_states).gather(1,next_actions).squeeze(1)
                next_actions = self.policy_net_noisy(non_final_next_states).max(1)[1].unsqueeze(1)
                next_q_values[non_final_mask] = self.target_net_noisy(non_final_next_states).gather(1,next_actions).squeeze(1)
        # 计算下一个状态的最大Q值
        # with torch.no_grad():
        #     # next_q_values = self.target_net(next_state_batch).max(1)[0]
        #     next_q_values = self.target_net_mlp(next_state_batch).max(1)[0]

        # 计算期望Q值
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(1), expected_q_values)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.update_count += 1

        # 软更新目标网络
        # if self.update_ count % 100 == 0:
        self._soft_update_target_net()

        return loss.item()

    def get_avg_sigma(self):
        """
        计算当前 policy_net 中所有 NoisyLinear 层的 weight_sigma 和 bias_sigma 的平均绝对值
        """
        sigmas = []
        for module in self.policy_net_noisy.modules():
            if hasattr(module, 'weight_sigma') and hasattr(module, 'bias_sigma'):
                sigmas.append(module.weight_sigma.abs().mean().item())
                sigmas.append(module.bias_sigma.abs().mean().item())
        return np.mean(sigmas) if sigmas else 0.0


    def _soft_update_target_net(self):
        """软更新目标网络参数"""
        for target_param, policy_param in zip(self.target_net_noisy.parameters(), self.policy_net_noisy.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net_noisy.state_dict(),
            'target_net_state_dict': self.target_net_noisy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"模型已保存到 {path}")

    def load_model(self, path):
        """加载模型"""
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            return False

        checkpoint = torch.load(path)
        self.policy_net_noisy.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net_noisy.load_state_dict(checkpoint['target_net_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"模型已从 {path} 加载")
        return True
