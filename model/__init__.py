import torch
from torch import nn
import numpy as np

from yet_another_retnet.retnet import RetNet


class ReinforcementNet(nn.Module):
    def __init__(self, input_shape, embedding_size=64):
        total_input_size = np.prod(input_shape)
        self.represent = nn.Sequential(
            nn.Linear(total_input_size, embedding_size),
            nn.LeakyReLU()
        )
