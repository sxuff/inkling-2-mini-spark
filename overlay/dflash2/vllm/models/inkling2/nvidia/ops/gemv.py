# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for the inkling-2-mini-gb10 recipe (2026-11-27): bit-identical SM12x tiles
# (32x512 for lm_head, 16x512 for N>=4096, 16x256 for N<=1024, 32x256 otherwise).
"""Low-M BF16 tensor-core projections for SM120/SM121."""

import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

MAX_M = 16
_BN, _BK = 32, 256
_WIDE_N = 4096 if os.environ.get("INKLING2_GEMV_TILE", "1") != "0" else 1 << 62


@triton.jit
def _gemv_kernel(
    X,
    W,
    B,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def sm12x_bf16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    assert m <= MAX_M
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
    if m:
        if n >= 65536 and _WIDE_N <= 65536:
            bn, bk, stages = 32, 512, 3
        elif n >= _WIDE_N:
            bn, bk, stages = 16, 512, 3
        elif n <= 1024 and _WIDE_N <= 4096:
            bn, bk, stages = 16, 256, 2
        else:
            bn, bk, stages = _BN, _BK, 2 if _WIDE_N <= 4096 else 3
        _gemv_kernel[(triton.cdiv(n, bn),)](
            x2,
            weight,
            bias,
            out,
            m,
            n,
            k,
            x2.stride(0),
            weight.stride(0),
            out.stride(0),
            HAS_BIAS=bias is not None,
            BN=bn,
            BK=bk,
            MP=16,
            num_warps=4,
            num_stages=stages,
        )
    return out.view(*lead, n)


def use_sm12x_bf16_gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    return (
        x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim >= 2
        and weight.ndim == 2
        and x.shape[-1] == weight.shape[1]
        and x.stride(-1) == weight.stride(-1) == 1
        and 0 < x.numel() // x.shape[-1] <= MAX_M
        and (
            bias is None
            or (
                bias.dtype == torch.bfloat16
                and bias.device == x.device
                and bias.stride(0) == 1
            )
        )
        and current_platform.is_device_capability_family(120)
    )
