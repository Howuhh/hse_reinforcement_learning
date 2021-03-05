import gym
import random
import torch
import pybullet_envs  # Don't forget to install PyBullet!
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.distributions import Normal

ENV_NAME = "Walker2DBulletEnv-v0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

LAMBDA = 0.97
GAMMA = 0.99

ACTOR_LR = 3e-4
CRITIC_LR = 2e-4

CLIP = 0.2
ENTROPY_COEF = 1e-2
BATCHES_PER_UPDATE = 64
BATCH_SIZE = 64

MIN_TRANSITIONS_PER_UPDATE = 2048
MIN_EPISODES_PER_UPDATE = 4

ITERATIONS = 5000

def fix_seed(env, seed=0):
    env.seed(seed)
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    
def compute_lambda_returns_and_gae(trajectory):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_v = 0.
    
    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)
    
    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]
    

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim)
        )
        self.sigma = nn.Parameter(torch.zeros(action_dim))
        
    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions (use it to compute entropy loss) 
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand_as(mu)
        dist = Normal(mu, sigma)
        return torch.exp(dist.log_prob(action).sum(-1)), dist
        
    def act(self, state):
        # Returns an action, not-transformed action and distribution
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand_as(mu)
        dist = Normal(mu, sigma)
        pure_action = dist.sample()
        action = torch.tanh(pure_action)
        return action, pure_action, dist
        
    def loss(self, state, action, old_prob, advantage, eps):  # TODO: add entropy coef reduction
        new_prob, dist = self.compute_proba(state, action)
        entropy = dist.entropy().mean()
        
        clip_advantage = torch.clip(advantage, 1 - eps, 1 + eps) * advantage
        loss = -torch.min((new_prob / old_prob) * advantage, clip_advantage).mean() - ENTROPY_COEF * entropy
        
        return loss


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
        
    def get_value(self, state):
        return self.model(state)
    
    def loss(self, state, target_value):
        pred_value = self.get_value(state)
        
        return F.mse_loss(pred_value.view(-1), target_value)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic = Critic(state_dim).to(DEVICE)
        
        self.actor_optim = Adam(self.actor.parameters(), ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), CRITIC_LR)

    def update(self, trajectories):
        transitions = [t for traj in trajectories for t in traj] # Turn a list of trajectories into list of transitions
        state, action, old_prob, target_value, advantage = map(np.array, zip(*transitions))
        
        advnatage = (advantage - advantage.mean()) / (advantage.std() + 1e-16)      
        
        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE) # Choose random batch
            
            s = torch.tensor(state[idx], device=DEVICE).float()
            a = torch.tensor(action[idx], device=DEVICE).float()
            op = torch.tensor(old_prob[idx], device=DEVICE).float() # Probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx], device=DEVICE).float() # Estimated by lambda-returns 
            adv = torch.tensor(advantage[idx], device=DEVICE).float() # Estimated by generalized advantage estimation 
            
            # TODO: Update actor here   
            actor_loss = self.actor.loss(s, a, op, adv, CLIP)
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            # TODO: Update critic here
            critic_loss = self.critic.loss(s, v)
            
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            
            
    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), device=DEVICE).float()
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), device=DEVICE).float()
            action, pure_action, distr = self.actor.act(state)
            prob = torch.exp(distr.log_prob(pure_action).sum(-1))
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()

    def save(self, name="agent"):
        torch.save(self.actor, f"{name}.pkl")


def evaluate_policy(agent, episodes=5):
    env = gym.make(ENV_NAME)
    
    fix_seed(env, SEED)
    
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return np.mean(returns), np.std(returns)


def sample_episode(env, agent):
    state, done = env.reset(), False

    trajectory = []
    
    while not done:
        action, pure_action, prob = agent.act(state)
        value = agent.get_value(state)
        
        next_state, reward, done, _ = env.step(action)
        trajectory.append((state, pure_action, reward, prob, value))
        
        state = next_state
    
    return compute_lambda_returns_and_gae(trajectory)


def train():
    print("Traininig on: ", DEVICE)
    
    env = gym.make(ENV_NAME)
    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    
    fix_seed(env, SEED)
    
    state = env.reset()
    
    episodes_sampled, steps_sampled = 0, 0
    
    best_reward = -np.inf
    means, stds, iters = [], [], []
    
    for i in range(ITERATIONS):
        trajectories = []
        steps_ctn = 0
        
        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories)        
        
        if (i + 1) % (ITERATIONS // 100) == 0:
            mean, std = evaluate_policy(ppo, 50)
            print(f"Step: {i+1}, Reward mean: {mean}, Reward std: {std}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
            
            means.append(mean)
            stds.append(std)
            iters.append(steps_sampled)
            
            if mean > best_reward:
                best_reward = mean
                ppo.save(name="best_agent")
            ppo.save()
   
    return np.array(means), np.array(stds), np.array(iters)

if __name__ == "__main__":
    means, stds, iters = train()
    
    plt.figure(figsize=(12, 8))
    plt.plot(iters, means)
    plt.fill_between(iters, means - stds, means + stds, alpha=0.3)
    plt.xlabel("Transitions")
    plt.ylabel("Mean reward")
    plt.savefig('PPO.png', bbox_inches='tight')
