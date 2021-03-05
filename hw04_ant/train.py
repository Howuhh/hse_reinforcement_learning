import os
import gym
import torch
import random
import pybullet_envs

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Normal

from copy import deepcopy
from collections import deque


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(env, seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)
    

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        
    def add(self, transition):
        self.buffer.append(transition)
    
    def sample(self, size):
        batch = random.sample(self.buffer, size)
        return list(zip(*batch))
    

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        
        self.action_size = action_size
        
        self.hidden = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 2 * action_size)
        )
        
    def forward(self, state, return_logprob=False):  # TODO: add determenistic actions
        mu, log_sigma = torch.split(self.hidden(state), self.action_size, -1)
        
        policy_dist = Normal(mu, torch.exp(log_sigma))
        action = policy_dist.rsample() # reparameterization trick
        
        tanh_action = torch.tanh(action)
        
        if return_logprob:
            # log prob for trick with rsample + tanh (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(dim=-1) - torch.log(1 - tanh_action**2 + 1e-16).sum(dim=-1)

            return tanh_action, log_prob
        
        return tanh_action


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_size + action_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        
        return self.critic(state_action).view(-1)


class SoftActorCritic:
    def __init__(self, state_size, action_size, gamma=0.99, tau=0.005, alpha=0.2, actor_lr=1e-4, critic_lr=1e-4):
        self.actor = Actor(state_size, action_size).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic1 = Critic(state_size, action_size).to(DEVICE)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.target_critic1 = deepcopy(self.critic1)
        
        self.critic2 = Critic(state_size, action_size).to(DEVICE)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.target_critic2 = deepcopy(self.critic2)

        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)
            
    def _actor_loss(self, state):
        action, action_log_prob = self.actor(state, return_logprob=True)
        target_q = torch.min(
            self.critic1(state, action), 
            self.critic2(state, action)
        ) 
        loss = -(target_q - self.alpha * action_log_prob).mean()
        
        return loss

    def _critic_loss(self, state, action, next_state, reward, done):
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_state, return_logprob=True)
            
            Q_next = torch.min(
                self.target_critic1(next_state, next_action),
                self.target_critic2(next_state, next_action)
            )
            Q_target = reward + self.gamma * (1 - done) * (Q_next - self.alpha * next_action_log_prob) 
        
        Q1 = self.critic1(state, action)
        Q2 = self.critic2(state, action)
        
        loss = F.mse_loss(Q1, Q_target) + F.mse_loss(Q2, Q_target)
        
        return loss
        
    def update(self, batch):
        state, action, reward, next_state, done = batch
        
        state = torch.tensor(state, device=DEVICE, dtype=torch.float32)
        action = torch.tensor(action, device=DEVICE, dtype=torch.long)
        reward = torch.tensor(reward, device=DEVICE, dtype=torch.float32)
        next_state = torch.tensor(next_state, device=DEVICE, dtype=torch.float32)
        done = torch.tensor(done, device=DEVICE, dtype=torch.float32)
        
        critic_losses = self._critic_loss(state, action, next_state, reward, done)

        # Critic1 & Critic2 update
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_losses.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        actor_loss = self._actor_loss(state)
        
        # Actor update    
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #  Target networks soft update
        with torch.no_grad():
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, device=DEVICE, dtype=torch.float32)
            action = self.actor(state).cpu().numpy()
        return action
    
    def save(self, name):
        torch.save(self.actor, "agent.pkl")
        
        
def evaluate_policy(agent, episodes=5):
    env = gym.make("AntBulletEnv-v0")
    
    set_seed(env, SEED)
    
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    
    return np.mean(returns), np.std(returns)


def train(model, timesteps=500_000, start_train=10_000, buffer_size=100_000, batch_size=512, test_episodes=10, test_every=5000, update_every=10):
    print("Training on: ", DEVICE)
    env = gym.make("AntBulletEnv-v0")
    
    set_seed(env, SEED)
    
    buffer = ReplayBuffer(size=buffer_size)
    best_reward = -np.inf
    
    means, stds = [], []
    
    done, state = False, env.reset()
    
    for t in range(timesteps):
        if done:
            done, state = False, env.reset()
    
        if t >= start_train:
            action = model.act(state)
        else:
            action = env.action_space.sample()
        
        next_state, reward, done, _ = env.step(action)
        buffer.add((state, action, reward, next_state, int(done)))
    
        state = next_state
        
        if t >= start_train and t % update_every == 0:
            for _ in range(update_every):
                batch = buffer.sample(batch_size)
                model.update(batch)
            
            if (t + 1) % test_every == 0 or t == timesteps - 1:
                mean, std = evaluate_policy(model, episodes=test_episodes)
                print(f"Step: {t + 1}, Reward mean: {mean}, Reward std: {std}")
                
                if mean > best_reward:
                    best_reward = mean
                    model.save(f"best_agent_{mean}_{std}")
                
                model.save(f"last_agent_{mean}_{std}")
    
                means.append(mean)
                stds.append(std)
    
    return np.array(means), np.array(stds)

if __name__ == "__main__":
    sac = SoftActorCritic(state_size=28, action_size=8, gamma=0.99, tau=0.002, critic_lr=5e-4, actor_lr=2e-4)
    
    means, stds = train(sac, timesteps=1_000_000, start_train=10_000, buffer_size=200_000, batch_size=128, update_every=50, test_every=1_000)
