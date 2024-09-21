from typing import Any, Mapping
import torch.nn as nn
import torch

from flux_triton.ops.gelu import LigerGELUMulFunction


class LigerGELUMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.w1 = nn.Parameter(torch.ones(self.intermediate_size, self.hidden_size))
        self.b1 = nn.Parameter(torch.ones(self.intermediate_size))
        # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.w2 = nn.Parameter(torch.ones(self.hidden_size, self.intermediate_size))
        self.b2 = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, x):

        return LigerGELUMulFunction.apply(x, self.w1, self.b1, self.w2, self.b2)
        
    