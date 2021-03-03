import pybullet_envs
import gym
import random
import numpy as np
import os
import torch

from train import Actor

class Agent:
    def __init__(self, name=None):
        name = "/best_agent.pkl" if name is None else name

        self.model = torch.load(__file__[:-8] + name, map_location="cpu")
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            action, pure_action, distr = self.model.act(state)
        
        return action.cpu().numpy()[0]
            
    def reset(self):
        pass

def visualise_policy(agent, episodes=5):
    env = gym.make("Walker2DBulletEnv-v0")
    
    for _ in range(episodes):
        done = False
        
        env.render()
        state = env.reset()
                
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            env.camera_adjust()
            
if __name__ == "__main__":
    agent = Agent("best_agent.pkl")
    visualise_policy(agent, episodes=20)