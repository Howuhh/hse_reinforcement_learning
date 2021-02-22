import random
import numpy as np
import os
import torch

from train import DuelingQ  # import for torch.load


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "agent.pkl", map_location=torch.device('cpu'))
        self.model.eval()
        
    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device="cpu")
        action = torch.argmax(self.model(state)).cpu().numpy().item()
        return action

    def reset(self):
        pass

