import math
import operator

import torch
import triton
import triton.language as tl

from flux_triton.ops.utils import (
    calculate_settings,
    compare_version,
    ensure_contiguous,
)

if compare_version("triton", operator.ge, "3.0.0"):
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt


@triton.jit
def _layer_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_cols)
    Y_row_stride,  # stride of each row in output
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    X_row_stride,  # stride of each row in input
    scale,
    shift,
    _N,
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    Mean_row_stride,  # stride of each row in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    RSTD_row_stride,  # stride of each row in rstd
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    scale += (row_idx // _N) * X_row_stride
    shift += (row_idx // _N) * X_row_stride


    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
    var = tl.sum((X_row - mean) * (X_row - mean), axis=0) / n_cols
    rstd = rsqrt(var + eps)

    tl.store(Mean_ptr, mean)
    tl.store(RSTD_ptr, rstd)

    img_scale = tl.load(scale + col_offsets, mask=mask, other=0.).to(tl.float32)
    img_shift = tl.load(shift + col_offsets, mask=mask, other=0.).to(tl.float32)
    Y_row = (X_row - mean) * rstd
    Y_row = Y_row * (1.0 + img_scale) + img_shift

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


def layer_norm_forward(X, scale, shift, eps):
    Y = torch.empty_like(X)
    N = X.shape[1]
    X = X.reshape(-1, X.shape[-1])

    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    Mean = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=X.dtype, device=X.device)

    _layer_norm_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        scale,
        shift,
        N,
        Mean,
        Mean.stride(0),
        RSTD,
        RSTD.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y.view(*shape), X, Mean, RSTD, BLOCK_SIZE, num_warps


class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, scale, shift, eps):
        Y, X, Mean, RSTD, BLOCK_SIZE, num_warps = layer_norm_forward(X, scale, shift, eps)
        ctx.save_for_backward(X, Mean, RSTD)
        return Y