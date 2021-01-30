import os
import gym
import random
import numpy as np

import matplotlib.pyplot as plt

try:
    from utils import plot_learning_curve
except ModuleNotFoundError:
    from .utils import plot_learning_curve

GAMMA = 0.98
GRID_SIZE_X = 30 # default 30
GRID_SIZE_Y = 30 # default 30
EPS = 0.1

# Num    Observation               Min            Max
# 0      Car Position              -1.2           0.6
# 1      Car Velocity              -0.07          0.07
# Simple discretization: map to [0, 1] then to {0, 1, .... ,(x_dim * y_dim) - 1}
def transform_state(state):
    assert len(state) == 2, "wrong state dim"
    
    state = (np.array(state) + np.array((1.2, 0.07))) / np.array((1.8, 0.14))
    x = min(int(state[0] * GRID_SIZE_X), GRID_SIZE_X - 1)
    y = min(int(state[1] * GRID_SIZE_Y), GRID_SIZE_Y - 1)
    return x + GRID_SIZE_X*y


class QLearning:
    def __init__(self, state_dim, action_dim, alpha=0.1):
        self.Q_table = np.zeros((state_dim, action_dim)) + 2.0
        self.alpha = alpha

    def update(self, transition):
        state, action, next_state, reward, done = transition
        
        if done:
            self.Q_table[next_state, :] = 0
        
        self.Q_table[state, action] += self.alpha * (reward + GAMMA * np.max(self.Q_table[next_state]) - self.Q_table[state, action])

    def act(self, state):
        return np.argmax(self.Q_table[state])

    def save(self, path=""):
        np.savez(os.path.join(path, "q_agent.npz"), self.Q_table)

      
class SARSA:
    def __init__(self, state_dim, action_dim, alpha=0.1, eps=0.1):
        self.Q_table = np.zeros((state_dim, action_dim)) + 2.0
        self.alpha = alpha
        self.eps = eps
        
    def update(self, transition):    
        state, action, next_state, reward, done = transition
        
        next_action = self.act(next_state)
        
        if done:
            self.Q_table[next_state, :] = 0
            
        self.Q_table[state, action] += self.alpha * (reward + GAMMA * self.Q_table[next_state, next_action] - self.Q_table[state, action])
    
    def act(self, state):
        if random.random() < self.eps:
            action = np.random.choice(range(self.Q_table.shape[-1]))
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def save(self, path=""):
        np.savez(os.path.join(path, "sarsa_agent.npz"), self.Q_table)


def evaluate_policy(agent, episodes=5):
    env = gym.make("MountainCar-v0")
    
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.0
        
        while not done:
            action = agent.act(transform_state(state))
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)
    return returns


def train_sarsa():
    epochs = 30_000
    
    env = gym.make("MountainCar-v0")
    sarsa = SARSA(GRID_SIZE_X * GRID_SIZE_Y, 3, 0.1, EPS)
    
    reduction = EPS / epochs
    
    env.seed(42)
    random.seed(42)
    np.random.seed(42)
    
    log = [[], [], []]
    
    total_transitions = 0
    for epoch in range(epochs):
        done, old_state = False, env.reset()
        
        trajectory = []
        state = transform_state(old_state)
        
        while not done:
            action = sarsa.act(state)
            
            next_state, reward, done, _ = env.step(action)
            
            shaped_reward = reward + 300 * (GAMMA * abs(next_state[1]) - abs(old_state[1]))
            
            trajectory.append((state, action, transform_state(next_state), shaped_reward, done and next_state[0] > 0.5))
            
            state = transform_state(next_state)
            old_state = next_state
            total_transitions += 1
                    
        for transition in reversed(trajectory):
            sarsa.update(transition)
                
        if epoch % 100 == 0:
            rewards = evaluate_policy(sarsa, 10)
            
            print(f"Epoch {epoch} -- Total transitions {total_transitions} -- Reward {np.mean(rewards)} +- {np.std(rewards)}") 
            
            log[0].append(total_transitions)
            log[1].append(np.mean(rewards))
            log[2].append(np.std(rewards))
            
        if sarsa.eps > 0:
            sarsa.eps = sarsa.eps - reduction
            
    sarsa.save()
    
    plot_learning_curve(*map(np.array, log), label="sarsa")
    

def train_q():
    epochs = 30_000
    eps = EPS
    reduction = eps / epochs
    
    env = gym.make("MountainCar-v0")
    Q = QLearning(GRID_SIZE_X * GRID_SIZE_Y, 3, 0.1)
    
    env.seed(42)
    random.seed(42)
    np.random.seed(42)
    
    log = [[], [], []]
    
    total_transitions = 0
    for epoch in range(epochs):
        done, old_state = False, env.reset()
        
        trajectory = []
        state = transform_state(old_state)
        
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = Q.act(state)
            
            next_state, reward, done, _ = env.step(action)
            
            shaped_reward = reward + 300 * (GAMMA * abs(next_state[1]) - abs(old_state[1]))
            
            trajectory.append((state, action, transform_state(next_state), shaped_reward, done and next_state[0] > 0.5))
            
            state = transform_state(next_state)
            old_state = next_state
            total_transitions += 1
                    
        for transition in reversed(trajectory):
            Q.update(transition)
                
        if epoch % 100 == 0:
            rewards = evaluate_policy(Q, 10)
            
            print(f"Epoch {epoch} -- Total transitions {total_transitions} -- Reward {np.mean(rewards)} +- {np.std(rewards)} -- Eps {eps}") 
            
            log[0].append(total_transitions)
            log[1].append(np.mean(rewards))
            log[2].append(np.std(rewards))
            
        if eps > 0:
            eps = eps - reduction
            
    Q.save()
    
    plot_learning_curve(*map(np.array, log), label="q-learning")


if __name__ == "__main__":
    plt.figure(figsize=(12, 8))
    train_sarsa()
    train_q()
    # agent = QLearning(30*30, 3, 0.1)
    # agent.Q_table = np.load("q_agent.npz")['arr_0']
    
    # print(np.mean(evaluate_policy(agent)))
    
