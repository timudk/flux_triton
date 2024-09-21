import torch
from einops import rearrange
from torch import Tensor
import os
from flux_triton.modules.rope import liger_rotary_pos_emb
from flash_attn_interface import flash_attn_func

def fa3(q, k, v):
    q, k, v = [t.permute(0,2,1,3) for t in [q,k,v]]
    out = flash_attn_func(q,k,v)[0]
    return out.permute(0,2,1,3)

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    triton_rope = os.getenv("TRITON_ROPE")
    use_fa3 = os.getenv("FA3")

    if triton_rope:
        cos = pe[:, 0, :, :, 0, :].reshape(pe.shape[0], pe.shape[2], -1)
        sin = pe[:, 0, :, :, 1, :].reshape(pe.shape[0], pe.shape[2], -1)
        q, k = liger_rotary_pos_emb(q, k, cos, sin)
    else:
        q, k = apply_rope(q, k, pe)

    if use_fa3:
        x = fa3(q, k, v)
    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
