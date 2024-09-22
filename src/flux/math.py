import torch
from einops import rearrange
from torch import Tensor
import os
from flux_triton.modules.rope import liger_rotary_pos_emb

if os.getenv("FA3"):
    from flash_attn_interface import flash_attn_func

    def fa3(q, k, v):
        q, k, v = [t.permute(0,2,1,3) for t in [q,k,v]]
        out = flash_attn_func(q,k,v)[0]
        return out.permute(0,2,1,3)

def attention(q: Tensor, k: Tensor, v: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    triton_rope = os.getenv("TRITON_ROPE")
    use_fa3 = os.getenv("FA3")

    if triton_rope:
        raise NotImplementedError("Currently not working")
        cos = pe[:, 0, :, :, 0, :].reshape(pe.shape[0], pe.shape[2], -1)
        sin = pe[:, 0, :, :, 1, :].reshape(pe.shape[0], pe.shape[2], -1)
        q, k = liger_rotary_pos_emb(q, k, cos, sin)
    else:
        q, k = apply_rope(q, k, cos, sin)

    if use_fa3:
        x = fa3(q, k, v)
    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> tuple[Tensor, Tensor]:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)

    return torch.cos(out).float(), torch.sin(out).float()


def apply_rope(xq: Tensor, xk: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    # xq, xk: [b, h, n, d]
    # cos, sin: [b, n, d // 2]

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    xq_1, xq_2 = rearrange(xq.float(), "... (d k) -> k ... d 1", k=2)
    xk_1, xk_2 = rearrange(xk.float(), "... (d k) -> k ... d 1", k=2)

    xq_out = torch.stack((cos, sin), dim=-1) * xq_1 + torch.stack((-sin, cos), dim=-1) * xq_2
    xk_out = torch.stack((cos, sin), dim=-1) * xk_1 + torch.stack((-sin, cos), dim=-1) * xk_2
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
