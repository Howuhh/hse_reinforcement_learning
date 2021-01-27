import os
import random
import numpy as np


class Agent:
    def __init__(self):
        self.qlearning_estimate = np.load(__file__[:-8] + "/agent.npz")
        
    def act(self, state):
        return 0

    def reset(self):
        pass

