import os
import gym
import torch
import random
import pybullet_envs
import numpy as np


try:
    from .train import Actor
except ModuleNotFoundError:
    from train import Actor


class Agent:
    def __init__(self, path=None):
        self.path = "/best_agent.pkl" if path is None else path
        
        self.model = Actor(state_size=28, action_size=8)
        self.model.load_state_dict(torch.load(__file__[:-8] + self.path, map_location="cpu"))
        
    def act(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32)
        action = self.model(state, eval_mode=True)
        
        return action.detach().numpy()

    def reset(self):
        pass


def visualise_policy(agent, episodes=5):
    env = gym.make("AntBulletEnv-v0")
    
    for _ in range(episodes):
        done = False
        
        env.render()
        state = env.reset()
                
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            env.camera_adjust()


if __name__ == "__main__":
    agent = Agent("best_agent.pkl")
    
    visualise_policy(agent, episodes=10)