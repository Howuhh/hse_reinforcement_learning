import os
import gym
import torch
import random
import itertools
import pybullet_envs

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Normal

from copy import deepcopy
from collections import deque

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(env, seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        
    def add(self, transition):
        self.buffer.append(transition)
    
    def sample(self, size):
        batch = random.sample(self.buffer, size)
        return list(zip(*batch))


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, act_limit=1.0):
        super().__init__()
        
        self.act_limit = act_limit
        self.action_size = action_size
        
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * action_size)
        )
        
    def forward(self, state, eval_mode=False, return_logprob=False):
        mu, log_sigma = torch.split(self.actor(state), self.action_size, -1)

        # from SAC paper 
        LOG_STD_MAX = 2
        LOG_STD_MIN = -20
        
        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        policy_dist = Normal(mu, torch.exp(log_sigma))
        
        if eval_mode:
            action = mu
        else:
            action = policy_dist.rsample() # reparameterization trick
        
        tanh_action = torch.tanh(action)
        
        if return_logprob:
            # log prob for trick with rsample + tanh (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1) - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

            return tanh_action * self.act_limit, log_prob
        
        return tanh_action * self.act_limit


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        
        return self.critic(state_action).view(-1) 


class SoftActorCritic:
    def __init__(self, state_size, action_size, hidden_size, gamma=0.99, tau=0.005, init_alpha=None, actor_lr=1e-4, critic_lr=1e-4, alpha_lr=1e-4):
        self.actor = Actor(state_size, action_size, hidden_size).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic1 = Critic(state_size, action_size, hidden_size).to(DEVICE)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.target_critic1 = deepcopy(self.critic1)
        
        self.critic2 = Critic(state_size, action_size, hidden_size).to(DEVICE)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.target_critic2 = deepcopy(self.critic2)

        for p in itertools.chain(self.target_critic1.parameters(), self.target_critic2.parameters()):
            p.requires_grad = False

        self.tau = tau
        self.gamma = gamma
        self.init_alpha = 0.0 if init_alpha is None else np.log(init_alpha)
        self.target_entropy = -float(action_size)
        
        self.log_alpha = torch.tensor([self.init_alpha], dtype=torch.float32, device=DEVICE, requires_grad=True)

        # self.log_alpha = torch.zeros(1, device=DEVICE, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp()

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    def _alpha_loss(self, state):
        with torch.no_grad():
            action, action_log_prob = self.actor(state, return_logprob=True)

        loss = (-self.alpha * (action_log_prob + self.target_entropy)).mean()

        return loss
            
    def _actor_loss(self, state):
        action, action_log_prob = self.actor(state, return_logprob=True)
        Q_target = torch.min(
            self.critic1(state, action), 
            self.critic2(state, action)
        )
        loss = (self.alpha.detach() * action_log_prob - Q_target).mean()
        
        assert action_log_prob.shape == Q_target.shape

        return loss

    def _critic_loss(self, state, action, reward, next_state, done):
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_state, return_logprob=True)
            
            Q_next = torch.min(
                self.target_critic1(next_state, next_action),
                self.target_critic2(next_state, next_action)
            )
            Q_target = reward + self.gamma * (1 - done) * (Q_next - self.alpha * next_action_log_prob) 

            assert Q_next.shape == next_action_log_prob.shape
        
        Q1 = self.critic1(state, action)
        Q2 = self.critic2(state, action)
        
        loss = F.mse_loss(Q1, Q_target) + F.mse_loss(Q2, Q_target)

        assert Q1.shape == Q_target.shape and Q2.shape == Q_target.shape
        
        return loss
        
    def update(self, batch):
        state, action, reward, next_state, done = batch
        
        state = torch.tensor(state, device=DEVICE, dtype=torch.float32)
        action = torch.tensor(action, device=DEVICE, dtype=torch.float32)
        reward = torch.tensor(reward, device=DEVICE, dtype=torch.float32)
        next_state = torch.tensor(next_state, device=DEVICE, dtype=torch.float32)
        done = torch.tensor(done, device=DEVICE, dtype=torch.float32)
        
        # Critic1 & Critic2 update
        critic_losses = self._critic_loss(state, action, reward, next_state, done)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_losses.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        # Actor update    
        actor_loss = self._actor_loss(state)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = self._alpha_loss(state)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()
        
        #  Target networks soft update
        with torch.no_grad():
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)

    def act(self, state, eval_mode=False):
        with torch.no_grad():
            state = torch.tensor(state, device=DEVICE, dtype=torch.float32)
            action = self.actor(state, eval_mode=eval_mode).cpu().numpy()
        return action
    
    def save(self, name):
        torch.save(self.actor, f"{name}.pkl")


def evaluate_policy(env_name, agent, seed, episodes=5):
    env = gym.make(env_name)
    
    set_seed(env, seed) # only env 
    
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state, eval_mode=True))
            total_reward += reward
        returns.append(total_reward)
    
    return np.mean(returns), np.std(returns)


def train(env_name, model, seed=0, timesteps=500_000, start_steps=10_000, start_train=1000, 
          buffer_size=100_000, batch_size=512, test_episodes=10, test_every=5000, update_every=10):
    print("Training on: ", DEVICE)
    
    env = gym.make(env_name)
    set_seed(env, seed)
    
    buffer = ReplayBuffer(size=buffer_size)
    best_reward = -np.inf
    
    means, stds = [], []
    
    done, state = False, env.reset()
    
    for t in range(timesteps):
        if done:
            done, state = False, env.reset()
    
        if t > start_steps:
            action = model.act(state)
        else:
            action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)        
        buffer.add((state, action, reward, next_state, done))
    
        state = next_state
        
        if t > start_train:
            if t % update_every == 0:
                for _ in range(update_every):
                    batch = buffer.sample(batch_size)
                    model.update(batch)
            
            if t % test_every == 0 or t == timesteps - 1:
                mean, std = evaluate_policy(env_name, model, seed=seed, episodes=test_episodes)
                print(f"Step: {t + 1}, Reward mean: {mean}, Reward std: {std}, Alpha: {model.alpha.detach().cpu().item()}")
                
                if mean > best_reward:
                    best_reward = mean
                    model.save(f"best_agent")
                
                model.save(f"last_agent")
    
                means.append(mean)
                stds.append(std)
    
    return np.array(means), np.array(stds)


if __name__ == "__main__":
    config = {
        "agent": {
            "state_size": 28,
            "action_size": 8,
            "hidden_size": 256,
            "gamma": 0.99,
            "tau": 0.002,
            "actor_lr": 2e-4,
            "critic_lr": 5e-4,
            "alpha_lr": 5e-5
        },
        "trainer": {
            "seed": 0,
            "timesteps": 3_000_000,
            "start_steps": 50_000,
            "start_train": 25_000,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "test_episodes": 10,
            "test_every": 10_000,
            "update_every": 16
        }
    }
    model = SoftActorCritic(**config["agent"])
    mean, std = train("AntBulletEnv-v0", model, **config["trainer"])
    
    np.save("means", mean)
    np.save("stds", std)
    
    
    