from typing import Any, Mapping
import torch.nn as nn

from flux_triton.ops.gelu import LigerGELUMulFunction


class LigerGELUMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)

    def forward(self, x):

        return self.down_proj(
            LigerGELUMulFunction.apply(self.up_proj(x), self.up_proj(x))
        )
    