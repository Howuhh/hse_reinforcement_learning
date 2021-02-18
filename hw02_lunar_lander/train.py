import gym
import torch
import copy
import random

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
from collections import deque


device = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        
    def add(self, transition):
        self.buffer.append(transition)
        
    def sample(self, size):
        batch = random.sample(self.buffer, size)
        return list(zip(*batch))
    

class DuelingQ(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.linear1 = nn.Linear(state_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        
        self.V = nn.Linear(32, 1)
        self.A = nn.Linear(32, action_dim)
        
    def forward(self, state):
        out = F.relu(self.linear1(state))
        out = F.relu(self.linear2(out))
        V, A = self.V(out), self.A(out)
        
        return V + (A - torch.mean(A, dim=-1, keepdims=True))


class DQN:
    def __init__(self, state_dim, action_dim, gamma, lr, double_q=False, dueling_q=False):
        
        if dueling_q:
            self.model = DuelingQ(state_dim, action_dim)
        else:
            self.model = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim)
            )
        
        self.model.to(device)
        self.target_model = deepcopy(self.model).to(device)
        
        self.gamma = gamma
        self.double_q = double_q
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def dqn_loss(self, state, action, reward, next_state, done):
        if self.double_q:
            next_action = torch.argmax(self.model(next_state), dim=1)
            Q_next = self.target_model(next_state)[torch.arange(len(next_action)), next_action]
        else:
            Q_next = self.target_model(next_state).max(dim=1).values
        
        Q_target = reward + self.gamma * (1 - done) * Q_next
        Q = self.model(state)[torch.arange(len(action)), action]
        
        return F.mse_loss(Q, Q_target)
        
    def update_target(self):
        self.target_model = deepcopy(self.model)
        self.target_model.to(device)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = torch.argmax(self.model(state)).cpu().numpy().item()
        return action

    def update(self, batch):
        state, action, reward, next_state, done = batch
        
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = torch.tensor(action, device=device, dtype=torch.long)
        reward = torch.tensor(reward, device=device, dtype=torch.float32)
        next_state = torch.tensor(next_state, device=device, dtype=torch.float32)
        done = torch.tensor(done, device=device, dtype=torch.float32)
        
        loss = self.dqn_loss(state, action, reward, next_state, done)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
        
    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = gym.make("LunarLander-v2")
    
    env.seed(0)
    
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.0
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return np.mean(returns), np.std(returns)


def train(model, timesteps=200_000, start_train=1000, batch_size=128, buffer_size=int(1e5), 
          update_policy=1, update_target=1000, eps_max=0.1, eps_min=0.0, test_every=5000):
    env = gym.make("LunarLander-v2")
    
    print("Training on: ", device)
    
    env.seed(0)
    env.action_space.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    
    rewards_total, stds_total = [], []
    loss_count, total_loss = 0, 0
    
    episodes = 0
    best_reward = -np.inf
    
    buffer = ReplayBuffer(size=buffer_size)
    
    done, state = False, env.reset()
    
    for step in range(timesteps):
        if done:
            done, state = False, env.reset()
            episodes += 1
        
        eps = eps_max - (eps_max - eps_min) * step / timesteps
        
        if random.random() < eps: 
            action = env.action_space.sample()
        else:
            action = model.act(state)

        next_state, reward, done, _ = env.step(action)
        buffer.add((state, action, reward, next_state, int(done)))
        
        state = next_state
        
        if step > start_train:
            if step % update_policy == 0:
                batch = buffer.sample(size=batch_size)
                loss = model.update(batch)
                
                total_loss += loss
                loss_count += 1
                
            if step % update_target == 0:
                model.update_target()
        
            if step % test_every == 0:
                mean, std = evaluate_policy(model, episodes=5)
                
                print(f"Episode: {episodes}, Step: {step + 1}, Reward mean: {mean}, Reward std: {std}, Loss: {total_loss / loss_count}")
                
                if mean > best_reward:
                    best_reward = mean
                    model.save()
                    
                rewards_total.append(mean)
                stds_total.append(std)
    
    return np.array(rewards_total), np.array(stds_total)


if __name__ == "__main__":
    model = DQN(8, 4, gamma=0.99, lr=1e-3, double_q=False, dueling_q=True)
    train(model, timesteps=200_000, start_train=10_000, update_policy=1, update_target=1000, batch_size=128, buffer_size=100_000, test_every=5000, eps_max=0.2)
 