import os
import random
import numpy as np

try:
    from train import transform_state
except ModuleNotFoundError:
    from .train import transform_state


class Agent:
    def __init__(self):
        self.qlearning_estimate = np.load(__file__[:-8] + "/q_agent_best.npz")['arr_0']
        
    def act(self, state):
        state = transform_state(state)
        return np.argmax(self.qlearning_estimate[state])
    
    def reset(self):
        pass

