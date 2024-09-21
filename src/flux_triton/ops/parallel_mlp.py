import torch
import time
import statistics
import torch.nn.functional as F

import math
import triton
import triton.language as tl

_kAlpha = math.sqrt(2 / math.pi)


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def gelu_forward(x):
    return 0.5 * x * (1 + tanh(_kAlpha * x * (1 + 0.044715 * x * x)))


@triton.jit
def fused_kernel(
    attn_ptr, mlp_ptr, weight_ptr, bias_ptr, x_ptr, output_ptr,
    M, N, K, C1, C2, gate,
    stride_am, stride_ak,
    stride_mm, stride_mk,
    stride_wk, stride_wn,
    stride_b,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Have a mask for the attention so we know where to load from
        # for the fused attention/mlp outputs.
        k_idx = k + offs_k
        mask_k = k_idx < K
        attn_mask = (k_idx < C1) & mask_k
        mlp_mask = (k_idx >= C1) & mask_k

        # this is just attention
        k_attn = k_idx
        a_ptrs_attn = attn_ptr + (offs_m[:, None] * stride_am + k_attn[None, :] * stride_ak)
        a_attn = tl.load(a_ptrs_attn, mask=mask_m[:, None] & attn_mask[None, :], other=0.0)

        # and this is the mlp + gelu
        k_mlp = k_idx - C1
        a_ptrs_mlp = mlp_ptr + (offs_m[:, None] * stride_mm + k_mlp[None, :] * stride_mk)
        mlp_vals = tl.load(a_ptrs_mlp, mask=mask_m[:, None] & mlp_mask[None, :], other=0.0)
        a_mlp = gelu_forward(mlp_vals)

        a = a_attn + a_mlp
        b_ptrs = weight_ptr + (k_idx[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)

    bias_ptrs = bias_ptr + offs_n * stride_b
    bias = tl.load(bias_ptrs, mask=mask_n, other=0.0)
    acc += bias[None, :]

    acc = acc * gate
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    x_vals = tl.load(x_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    acc += x_vals

    output_ptrs = output_ptr + (offs_m[:, None] * stride_om + offs_n[None, :])
    tl.store(output_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def fused_gelu_mlp(attn, mlp, linear2, x, gate):
    """
    result = x + gate * linear2(torch.cat((attn, GELU(mlp, approximate="tanh")), -1))
    """
    # TODO: tune these variables
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    M = attn.shape[0]
    C1 = attn.shape[1]
    C2 = mlp.shape[1]
    K = C1 + C2
    N = linear2.out_features

    weight = linear2.weight.T.contiguous()
    bias = linear2.bias.contiguous()

    attn = attn.contiguous()
    mlp = mlp.contiguous()
    x = x.contiguous()

    stride_am, stride_ak = attn.stride()
    stride_mm, stride_mk = mlp.stride()
    stride_wk, stride_wn = weight.stride()
    stride_b = bias.stride(0)
    stride_xm, stride_xn = x.stride()
    output = torch.empty_like(x)
    stride_om, stride_on = output.stride()

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch Triton kernel
    fused_kernel[grid](
        attn,
        mlp,
        weight,
        bias,
        x,
        output,
        M, N, K, C1, C2, gate,
        stride_am, stride_ak,
        stride_mm, stride_mk,
        stride_wk, stride_wn,
        stride_b,
        stride_xm, stride_xn,
        stride_om, stride_on,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return output
