import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 1. 简单的多层感知机网络 (MLP)
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 2. 经验回放池
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# 3. DQN 智能体
class DQNAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.95, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 主网络和目标网络
        self.q_net = QNetwork(obs_dim, action_dim)
        self.target_net = QNetwork(obs_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64

    def select_action(self, obs, evaluate=False):
        # Epsilon-Greedy 探索策略
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # 计算当前 Q 值
        curr_Q = self.q_net(states).gather(1, actions)

        # 计算目标 Q 值
        with torch.no_grad():
            next_Q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_Q = rewards + (1 - dones) * self.gamma * next_Q

        # 计算损失并更新网络
        loss = nn.MSELoss()(curr_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)