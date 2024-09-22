import pytest
import torch
from flux_triton.modules.rope import liger_rotary_pos_emb_v2
from torch import Tensor
from einops import rearrange
SLEEP_SECONDS = 0.1

def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

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

def supports_bfloat16():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (8, 0)  # Ampere and newer

@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 128, 32, 32, 64),
        (2, 128, 32, 32, 64),
        # different q/k heads
        (1, 128, 32, 8, 64),
        (2, 128, 32, 8, 64),
        # weird shapes
        # HuggingFace llama/mistral source code doesn't support odd head dimension
        # so we don't test it here
        (3, 423, 73, 213, 92),
        (3, 423, 73, 155, 92),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-5,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
    ],
)
def test_correctness(
    bsz, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, atol, rtol
):
    _tensor_q = (
        torch.randn((bsz, seq_len, num_q_heads, head_dim), device="cuda")
        .transpose(1, 2)
        .to(dtype)
    )

    _tensor_k = (
        torch.randn((bsz, seq_len, num_kv_heads, head_dim), device="cuda")
        .transpose(1, 2)
        .to(dtype)
    )

    q1 = _tensor_q.clone().requires_grad_(True)
    k1 = _tensor_k.clone().requires_grad_(True)

    q2 = _tensor_q.clone().requires_grad_(True)
    k2 = _tensor_k.clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0)
    pe = rope(pos_ids, head_dim, 10000)

    
    q, k = apply_rope(q1, k1, pe)

    # validate forward pass
    # hf_q, hf_k = apply_rotary_pos_emb(q1, k1, cos, sin, pos_ids)
    tt_q, tt_k = liger_rotary_pos_emb_v2(q2, k2, pe)
    assert torch.allclose(q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(k, tt_k, atol=atol, rtol=rtol)

    # # validate backward pass
    # dq, dk = (
    #     torch.randn_like(hf_q, device="cuda"),
    #     torch.randn_like(hf_k, device="cuda").to(dtype),
    # )

    # q1_grad, k1_grad = torch.autograd.grad(
    #     (hf_q, hf_k), (q1, k1), (dq, dk), allow_unused=True
    # )
    # q2_grad, k2_grad = torch.autograd.grad(
    #     (tt_q, tt_k), (q2, k2), (dq.clone(), dk.clone()), allow_unused=True
    # )

    # assert torch.allclose(q1_grad, q2_grad, atol=atol, rtol=rtol)
    # assert torch.allclose(k1_grad, k2_grad, atol=atol, rtol=rtol)

if __name__ == "__main__":
    test_correctness(2, 128, 32, 32, 64, torch.float32, 1e-1, 1e-5)