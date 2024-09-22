import torch.nn as nn

from flux_triton.ops.layer_norm import LigerLayerNormFunction


class LigerLayerNorm(nn.Module):
    def __init__(self, hidden_size, elementwise_affine=False, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        assert not elementwise_affine

    def forward(self, hidden_states, scale, shift):
        return LigerLayerNormFunction.apply(
            hidden_states, scale, shift, self.eps
        )

    def extra_repr(self):
        return f"{self.hidden_size}, eps={self.eps}"