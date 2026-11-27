# SPDX-License-Identifier: Apache-2.0
# Modified for the inkling-2-mini-spark recipe (2026-11-27): new file, INT4/INT8 low-M GEMVs for DFlash2 draft-only weights.
"""Weight-only 4-bit (asymmetric per-group INT4) and 8-bit (per-row INT8)
low-M projections for SM12x DFlash2 drafts, registered as custom torch ops
for full CUDA graph and torch.compile compatibility.

All quantized weights are draft-only copies; the target model verifies every
draft token with untouched weights so output exactness is strictly preserved.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .gemv_q8 import MAX_M, gemv_q8, quantize_rows

__all__ = [
    "MAX_M",
    "quantize_rows",
    "quantize_int4_group",
    "draft_gemv_q8",
    "draft_gemv_q4",
]


@torch.no_grad()
def quantize_int4_group(
    weight: torch.Tensor,
    group_size: int = 64,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    chunk: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric per-group 4-bit quantization of bf16 `weight` [N, K].

    Pairs of consecutive groups (2*p, 2*p+1) are packed into uint8 bytes
    (lo nibble = even group, hi nibble = odd group) so each Triton tile loads
    contiguous packed bytes without bit-interleaving overhead.
    """
    n, k = weight.shape
    g = group_size
    assert k % (2 * g) == 0, f"K={k} must be divisible by 2*G={2 * g}"
    num_groups = k // g
    if out is None:
        out = (
            torch.empty((n, k // 2), dtype=torch.uint8, device=weight.device),
            torch.empty((num_groups, n), dtype=torch.bfloat16, device=weight.device),
            torch.empty((num_groups, n), dtype=torch.bfloat16, device=weight.device),
        )
    q_packed, scales, zeros = out
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        w = weight[start:end].float().view(end - start, num_groups, g)
        wmin = w.amin(dim=-1)
        wmax = w.amax(dim=-1)
        s = ((wmax - wmin) / 15.0).clamp_min(1e-12)
        q = (
            ((w - wmin.unsqueeze(-1)) / s.unsqueeze(-1))
            .round()
            .clamp_(0, 15)
            .to(torch.uint8)
        )
        q_pairs = q.view(end - start, num_groups // 2, 2, g)
        packed = (q_pairs[:, :, 0, :] | (q_pairs[:, :, 1, :] << 4)).reshape(
            end - start, k // 2
        )
        q_packed[start:end].copy_(packed)
        scales[:, start:end].copy_(s.to(torch.bfloat16).T)
        zeros[:, start:end].copy_(wmin.to(torch.bfloat16).T)
    return q_packed, scales, zeros


@triton.jit
def _gemv_q4_kernel(
    X,
    W,
    S,
    Z,
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
    G: tl.constexpr,
    MP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    cols = pid_n * BN + tl.arange(0, BN)
    rows = pid_m * MP + tl.arange(0, MP)
    col_mask = cols < N
    row_mask = rows < M
    ks_g = tl.arange(0, G)
    acc = tl.zeros((MP, BN), tl.float32)
    num_pairs = K // (2 * G)
    for p in range(0, num_pairs):
        k_lo = p * (2 * G) + ks_g
        k_hi = k_lo + G
        byte_off = p * G + ks_g
        packed = tl.load(
            W + cols[None, :].to(tl.int64) * SW + byte_off[:, None],
            col_mask[None, :],
            other=0,
        )
        g_lo = p * 2
        g_hi = g_lo + 1
        s_lo = tl.load(S + g_lo * N + cols, col_mask, other=0.0)
        z_lo = tl.load(Z + g_lo * N + cols, col_mask, other=0.0)
        s_hi = tl.load(S + g_hi * N + cols, col_mask, other=0.0)
        z_hi = tl.load(Z + g_hi * N + cols, col_mask, other=0.0)

        w_lo = (packed & 0x0F).to(tl.bfloat16) * s_lo[None, :] + z_lo[None, :]
        w_hi = (packed >> 4).to(tl.bfloat16) * s_hi[None, :] + z_hi[None, :]

        x_lo = tl.load(
            X + rows[:, None] * SX + k_lo[None, :],
            row_mask[:, None],
            other=0.0,
        )
        x_hi = tl.load(
            X + rows[:, None] * SX + k_hi[None, :],
            row_mask[:, None],
            other=0.0,
        )
        acc += tl.dot(x_lo, w_lo)
        acc += tl.dot(x_hi, w_hi)

    if HAS_BIAS:
        acc += tl.load(B + cols, col_mask, other=0.0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        row_mask[:, None] & col_mask[None, :],
    )


def _gemv_q4_impl(
    x: torch.Tensor,
    q_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m, k = x2.shape
    n = q_packed.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m:
        mp = 16 if m <= 16 else 32 if m <= 32 else 64
        bn = 16 if n <= 4096 else 32
        stages = 3 if (mp == 16 and group_size <= 64) else 2
        _gemv_q4_kernel[(triton.cdiv(n, bn), triton.cdiv(m, mp))](
            x2,
            q_packed,
            scales,
            zeros,
            bias if bias is not None else out,
            out,
            m,
            n,
            k,
            x2.stride(0),
            q_packed.stride(0),
            out.stride(0),
            HAS_BIAS=bias is not None,
            BN=bn,
            G=group_size,
            MP=mp,
            num_warps=4,
            num_stages=stages,
        )
    return out.view(*lead, n)


def _gemv_q4_fake(
    x: torch.Tensor,
    q_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], q_packed.shape[0]))


direct_register_custom_op(
    op_name="inkling2_draft_gemv_q4",
    op_func=_gemv_q4_impl,
    fake_impl=_gemv_q4_fake,
)


def _gemv_q8_impl(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return gemv_q8(x, q, scale, bias=bias)


def _gemv_q8_fake(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], q.shape[0]))


direct_register_custom_op(
    op_name="inkling2_draft_gemv_q8",
    op_func=_gemv_q8_impl,
    fake_impl=_gemv_q8_fake,
)


def draft_gemv_q8(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.inkling2_draft_gemv_q8(x, q, scale, bias)


def draft_gemv_q4(
    x: torch.Tensor,
    q_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 64,
) -> torch.Tensor:
    return torch.ops.vllm.inkling2_draft_gemv_q4(
        x, q_packed, scales, zeros, bias, group_size
    )
