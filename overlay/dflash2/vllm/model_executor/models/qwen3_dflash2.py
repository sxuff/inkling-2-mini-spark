# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for the inkling-2-mini-spark recipe (2026-11-27): draft-only INT8/INT4/NVFP4 weight copies (DFLASH2_HEAD_QUANT, DFLASH2_BODY_QUANT), fused conv+residual+norm (DFLASH2_FUSED_OPS), selector softcap placement, embed_norm.

import os

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker.gpu.spec_decode.dflash2.config import DFlash2SpeculatorConfig

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import maybe_prefix


@triton.jit
def _dflash2_grouped_conv_kernel(
    x_ptr,
    delta_ptr,
    base_ptr,
    output_ptr,
    x_stride_row,
    delta_stride_row,
    delta_stride_tap,
    base_stride_tap,
    output_stride_row,
    NUM_CHANNELS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TAPS: tl.constexpr,
    ELEMENT_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0) // triton.cdiv(NUM_CHANNELS, ELEMENT_BLOCK)
    col_block = tl.program_id(0) % triton.cdiv(NUM_CHANNELS, ELEMENT_BLOCK)
    channels = col_block * ELEMENT_BLOCK + tl.arange(0, ELEMENT_BLOCK)
    mask = channels < NUM_CHANNELS
    groups = channels // GROUP_SIZE
    position = row % BLOCK_SIZE

    delta_row = delta_ptr + row * delta_stride_row
    x_row = x_ptr + row * x_stride_row
    accumulator = (
        tl.load(base_ptr + channels, mask=mask, other=0.0).to(tl.float32)
        + tl.load(delta_row + groups, mask=mask, other=0.0).to(tl.float32)
    ) * tl.load(x_row + channels, mask=mask, other=0.0).to(tl.float32)

    for tap in tl.static_range(1, TAPS):
        tap_mask = mask & (position >= tap)
        coefficient = tl.load(
            base_ptr + tap * base_stride_tap + channels,
            mask=tap_mask,
            other=0.0,
        ).to(tl.float32) + tl.load(
            delta_row + tap * delta_stride_tap + groups,
            mask=tap_mask,
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            x_ptr + (row - tap) * x_stride_row + channels,
            mask=tap_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += coefficient * x

    tl.store(
        output_ptr + row * output_stride_row + channels,
        accumulator,
        mask=mask,
    )


def dflash2_grouped_conv_impl(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """DFlash2 grouped convolution adapted from LightSeek TokenSpeed PR #1399."""
    num_rows, num_channels = x.shape
    output = torch.empty_like(x)
    if num_rows == 0:
        return output

    element_block = 1024 if num_rows >= 128 and num_channels % 1024 == 0 else 512
    grid = (num_rows * triton.cdiv(num_channels, element_block),)
    _dflash2_grouped_conv_kernel[grid](
        x,
        delta,
        base,
        output,
        x.stride(0),
        delta.stride(0),
        delta.stride(1),
        base.stride(0),
        output.stride(0),
        NUM_CHANNELS=num_channels,
        BLOCK_SIZE=block_size,
        GROUP_SIZE=group_size,
        TAPS=base.shape[0],
        ELEMENT_BLOCK=element_block,
        num_warps=4,
    )
    return output


def dflash2_grouped_conv_fake(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="dflash2_grouped_conv",
    op_func=dflash2_grouped_conv_impl,
    fake_impl=dflash2_grouped_conv_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def dflash2_grouped_conv(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    return torch.ops.vllm.dflash2_grouped_conv(x, delta, base, block_size, group_size)


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    if hidden_states.is_cuda:
        return dflash2_grouped_conv(hidden_states, delta, base, block_size, group_size)

    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            # Query tokens per request: the bonus token plus the mask tokens.
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


@triton.jit
def _dflash2_score_edges_kernel(
    predecessor_ptr,
    successor_ptr,
    candidate_ptr,
    unary_ptr,
    hidden_ptr,
    anchor_ptr,
    output_ptr,
    anchor_stride,
    vocab_size,
    NUM_STEPS: tl.constexpr,
    TOP_K: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    POST_SOFTCAP: tl.constexpr,
):
    position = tl.program_id(0)
    req_idx = position // NUM_STEPS
    step = position % NUM_STEPS
    k = tl.arange(0, BLOCK_K)
    r = tl.arange(0, BLOCK_R)
    k_mask = k < TOP_K
    r_mask = r < RANK
    candidate_ids = tl.load(
        candidate_ptr + position * TOP_K + k, mask=k_mask, other=0
    ).to(tl.int64)
    if step == 0:
        # The predecessor for the first step is the anchor token.
        anchor = tl.load(anchor_ptr + req_idx * anchor_stride).to(tl.int64)
        anchor = tl.minimum(tl.maximum(0, anchor), vocab_size - 1)
        predecessor_ids = tl.zeros_like(candidate_ids) + anchor
    else:
        predecessor_ids = tl.load(
            candidate_ptr + (position - 1) * TOP_K + k, mask=k_mask, other=0
        ).to(tl.int64)

    unary = tl.load(unary_ptr + position * TOP_K + k, mask=k_mask, other=0.0)
    hidden = tl.load(hidden_ptr + position * RANK + r, mask=r_mask, other=0.0)
    mask = k_mask[:, None] & r_mask[None, :]
    predecessors = tl.load(
        predecessor_ptr + predecessor_ids[:, None] * RANK + r[None, :],
        mask=mask,
        other=0.0,
    )
    successors = tl.load(
        successor_ptr + candidate_ids[:, None] * RANK + r[None, :],
        mask=mask,
        other=0.0,
    )
    weighted = predecessors.to(tl.float32) * hidden[None, :].to(tl.float32)
    edges = tl.dot(weighted.to(successors.dtype), tl.trans(successors))
    val = unary[None, :].to(tl.float32) + edges.to(tl.float32)
    if POST_SOFTCAP > 0.0:
        z = val / POST_SOFTCAP
        ez = tl.exp(-2.0 * tl.abs(z))
        t = (1.0 - ez) / (1.0 + ez)
        val = POST_SOFTCAP * tl.where(z >= 0.0, t, -t)
    tl.store(
        output_ptr + (position * TOP_K + k[:, None]) * TOP_K + k[None, :],
        val,
        mask=k_mask[:, None] & k_mask[None, :],
    )


def _score_edges_output(candidate_ids: torch.Tensor) -> torch.Tensor:
    num_reqs, steps, top_k = candidate_ids.shape
    return candidate_ids.new_empty((num_reqs, steps, top_k, top_k), dtype=torch.float32)


def dflash2_score_edges_impl(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    post_softcap: float = 0.0,
) -> torch.Tensor:
    output = _score_edges_output(candidate_ids)
    num_reqs, steps, top_k = candidate_ids.shape
    if num_reqs * steps == 0:
        return output
    vocab_size, rank = predecessor_table.shape
    _dflash2_score_edges_kernel[(num_reqs * steps,)](
        predecessor_table,
        successor_table,
        candidate_ids.contiguous(),
        unary_logits.contiguous(),
        hidden.contiguous(),
        anchor_token_ids,
        output,
        anchor_token_ids.stride(0),
        vocab_size,
        NUM_STEPS=steps,
        TOP_K=top_k,
        RANK=rank,
        BLOCK_K=max(16, triton.next_power_of_2(top_k)),
        BLOCK_R=max(16, triton.next_power_of_2(rank)),
        POST_SOFTCAP=float(post_softcap),
        num_warps=4,
    )
    return output


def dflash2_score_edges_fake(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    post_softcap: float = 0.0,
) -> torch.Tensor:
    return _score_edges_output(candidate_ids)


direct_register_custom_op(
    op_name="dflash2_score_edges",
    op_func=dflash2_score_edges_impl,
    fake_impl=dflash2_score_edges_fake,
)


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
    post_softcap: float = 0.0,
) -> torch.Tensor:
    if candidate_ids.is_cuda:
        return torch.ops.vllm.dflash2_score_edges(
            predecessor_table,
            successor_table,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            float(post_softcap),
        )
    scores = _score_edges_reference(
        predecessor_table,
        successor_table,
        candidate_ids,
        unary_logits,
        hidden,
        anchor_token_ids,
        top_k,
    )
    if post_softcap > 0.0:
        scores = post_softcap * torch.tanh(scores / post_softcap)
    return scores


def _score_edges_reference(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    # A request still in chunked prefill anchors on its next prompt token, which
    # can lie past the draft vocabulary (an audio placeholder, say). Its draft is
    # discarded, so only keep the codebook lookup in range.
    anchor_token_ids = anchor_token_ids.clamp(0, predecessor_table.shape[0] - 1)
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits.float()[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


@support_torch_compile
class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        cap = float(getattr(self, "post_softcap", 0.0) or 0.0)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
            post_softcap=cap,
        )



def _parse_quant_mode(env_name: str, default: str = "off") -> str | None:
    val = os.environ.get(env_name, default).strip().lower()
    if val in ("", "0", "off", "none", "bf16"):
        return None
    if val not in ("int8", "int4_g64", "int4_g128", "nvfp4", "hybrid"):
        raise ValueError(
            f"{env_name}={val!r}: expected off, int8, int4_g64, int4_g128, nvfp4, or hybrid"
        )
    return val


class _MarlinNvFp4Holder(nn.Module):
    def __init__(self, packed: torch.Tensor, scales: torch.Tensor, gscale: torch.Tensor, n: int, k: int):
        super().__init__()
        self.weight = nn.Parameter(packed, requires_grad=False)
        self.weight_scale = nn.Parameter(scales, requires_grad=False)
        self.weight_global_scale = nn.Parameter(gscale, requires_grad=False)
        self.input_size_per_partition = k
        self.output_size_per_partition = n
        self.output_partition_sizes = [n]
        self.params_dtype = torch.bfloat16


def _quantize_draft_weight(weight: torch.Tensor, mode: str):
    from vllm.models.inkling2.nvidia.ops.gemv_q4 import (
        quantize_int4_group,
        quantize_rows,
    )

    if mode == "int8":
        return ("int8", quantize_rows(weight, "int8"))
    if mode == "int4_g64":
        return ("int4_g64", quantize_int4_group(weight, 64))
    if mode == "int4_g128":
        return ("int4_g128", quantize_int4_group(weight, 128))
    if mode == "nvfp4":
        from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
        from vllm.model_executor.layers.quantization.online.nvfp4 import (
            _quantize_nvfp4,
        )

        w = weight.contiguous()
        kernel = init_nvfp4_linear_kernel(use_a16=True, act_dtype=w.dtype)
        amax = w.abs().max().float().reshape(1)
        packed, scales, gscale = _quantize_nvfp4(w, amax)
        holder = _MarlinNvFp4Holder(packed, scales, gscale, w.shape[0], w.shape[1])
        kernel.process_weights_after_loading(holder)
        return ("nvfp4", (kernel, holder))
    raise ValueError(mode)


def _apply_draft_quant(
    x: torch.Tensor, q_entry: tuple, bias: torch.Tensor | None = None
) -> torch.Tensor:
    from vllm.models.inkling2.nvidia.ops.gemv_q4 import (
        draft_gemv_q4,
        draft_gemv_q8,
    )

    mode, tensors = q_entry
    if mode == "nvfp4":
        kernel, holder = tensors
        return kernel.apply_weights(holder, x, bias)
    if mode == "int8":
        q, scale = tensors
        return draft_gemv_q8(x, q, scale, bias)
    q_packed, scales, zeros = tensors
    g = 64 if mode == "int4_g64" else 128
    return draft_gemv_q4(x, q_packed, scales, zeros, bias, g)


class _DraftLowMQuantMethod:
    """Wraps a draft-only LinearBase quant_method to run INT8/INT4 Triton GEMV
    for low-M decode batches (M <= 64) and fall back to BF16 cuBLAS for prefill."""

    def __init__(self, orig_method, q_entry: tuple):
        self._orig = orig_method
        self._q_entry = q_entry

    def __getattr__(self, name):
        return getattr(self._orig, name)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and 0 < x.numel() // x.shape[-1] <= 64
        ):
            return _apply_draft_quant(x, self._q_entry, bias)
        return self._orig.apply(layer, x, bias)


class DFlash2Qwen3Model(DFlashQwen3Model):
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        self._fused_kv_q = None
        self._use_fused_ops = os.environ.get("DFLASH2_FUSED_OPS", "0") == "1"
        # Without its own tag the selector shares the draft head's compile cache.
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = super().embed_input_ids(input_ids)
        embed_norm = getattr(self, "embed_norm", None)
        if embed_norm is not None:
            embeds = embed_norm(embeds)
        return embeds * self.input_embedding_scale

    def _project_context_kv_token_major(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        if (
            self._fused_kv_q is not None
            and context_states.is_cuda
            and context_states.dtype == torch.bfloat16
            and 0 < num_ctx <= 64
        ):
            normed = self._normalize_context_states(context_states)
            all_kv_flat = _apply_draft_quant(
                normed, self._fused_kv_q, self._fused_kv_bias
            )
            return all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
        return super()._project_context_kv_token_major(
            context_states, num_ctx, num_layers, num_kv_heads, head_dim
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._use_fused_ops:
            return super().forward(input_ids, positions, input_embeds)
        from vllm.models.beam.nvidia.dflash2_conv_ops import conv_residual_norm

        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        residual = input_embeds
        hidden_states = self.layers[0].input_layernorm(input_embeds)
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            next_norm = (
                self.layers[i + 1].input_layernorm if i + 1 < n_layers else self.norm
            )
            conv = layer.attention_conv
            hidden_states, coefficients = conv.prepare(hidden_states)
            hidden_states = layer.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states, residual = conv_residual_norm(
                hidden_states,
                coefficients,
                conv.base_kernel[1],
                residual,
                layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.variance_epsilon,
                conv.block_size,
                conv.group_size,
            )
            conv = layer.mlp_conv
            hidden_states, coefficients = conv.prepare(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states, residual = conv_residual_norm(
                hidden_states,
                coefficients,
                conv.base_kernel[1],
                residual,
                next_norm.weight,
                next_norm.variance_epsilon,
                conv.block_size,
                conv.group_size,
            )
        return hidden_states

    @torch.no_grad()
    def prepare_body_quant(self) -> None:
        mode = _parse_quant_mode("DFLASH2_BODY_QUANT", "off")
        if mode is None:
            return
        quant_fc = os.environ.get("DFLASH2_QUANT_FC", "1") == "1"
        def _min_k(m_mode: str) -> int:
            if m_mode == "int4_g64":
                return 128
            if m_mode == "int4_g128":
                return 256
            if m_mode == "nvfp4":
                return 64
            return 1

        def _wrap(mod: nn.Module, m_mode: str) -> None:
            w = getattr(mod, "weight", None)
            qm = getattr(mod, "quant_method", None)
            if w is None or qm is None or w.dtype != torch.bfloat16 or not w.is_cuda:
                return
            if w.ndim != 2 or w.shape[1] % _min_k(m_mode) != 0:
                return
            if m_mode == "nvfp4" and (w.shape[0] % 64 != 0):
                m_mode = "int8"
            q_entry = _quantize_draft_weight(w, m_mode)
            mod.quant_method = _DraftLowMQuantMethod(qm, q_entry)

        if quant_fc and hasattr(self, "fc"):
            fc_mode = "int8" if mode in ("int4_g64", "int4_g128", "nvfp4", "hybrid") else mode
            _wrap(self.fc, fc_mode)

        kv_mode = "int8" if mode == "hybrid" else mode
        if self._fused_kv_weight is not None and self._fused_kv_weight.is_cuda:
            if self._fused_kv_weight.shape[1] % _min_k(kv_mode) == 0:
                self._fused_kv_q = _quantize_draft_weight(self._fused_kv_weight, kv_mode)

        for layer in self.layers:
            if mode == "hybrid":
                _wrap(layer.self_attn.qkv_proj, "int8")
                _wrap(layer.self_attn.o_proj, "int8")
                _wrap(layer.mlp.gate_up_proj, "nvfp4")
                _wrap(layer.mlp.down_proj, "nvfp4")
            else:
                _wrap(layer.self_attn.qkv_proj, mode)
                _wrap(layer.self_attn.o_proj, mode)
                _wrap(layer.mlp.gate_up_proj, mode)
                _wrap(layer.mlp.down_proj, mode)
            _wrap(layer.attention_conv.kernel_projection, "int8")
            _wrap(layer.mlp_conv.kernel_projection, "int8")
        _wrap(self.candidate_selector.hidden_projection, "int8")


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    # The speculator this is a draft model of (see SpeculativeDraftModel).
    speculator_config_class = DFlash2SpeculatorConfig

    model_cls = DFlash2Qwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        draft_config = self.config.dflash_config
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        softcap_mode = draft_config.get("selector_softcap_mode", "before_correction")
        self._cand_vocab_size = min(
            vllm_config.model_config.get_vocab_size(), self.config.vocab_size
        )
        self._cand_scale = float(draft_config.get("output_multiplier", 1.0))
        self._cand_pre_softcap = (
            softcap if softcap > 0 and softcap_mode == "before_correction" else None
        )
        # Candidates index the selector codebooks, which have the draft's
        # vocabulary; the target's vocabulary can be larger.
        self.candidate_logits_processor = LogitsProcessor(
            self._cand_vocab_size,
            scale=self._cand_scale,
            soft_cap=self._cand_pre_softcap,
        )
        self.model.candidate_selector.post_softcap = (
            softcap if softcap > 0 and softcap_mode == "after_correction" else 0.0
        )
        self._draft_lm_head_q = None

    @torch.no_grad()
    def prepare_draft_quant(self) -> None:
        self.model.prepare_body_quant()
        head_mode = _parse_quant_mode("DFLASH2_HEAD_QUANT", "off")
        if head_mode is not None and self.lm_head is not None:
            w = self.lm_head.weight
            if w.is_cuda and w.dtype == torch.bfloat16:
                # Slice to unpadded vocab rows so we skip computing padded rows.
                n_rows = min(w.shape[0], self._cand_vocab_size)
                self._draft_lm_head_q = _quantize_draft_weight(
                    w[:n_rows].contiguous(), head_mode
                )

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._draft_lm_head_q is not None
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and 0 < hidden_states.shape[0] <= 64
            and getattr(self.lm_head, "tp_size", 1) == 1
        ):
            from vllm.model_executor.layers.logits_processor import _topk

            logits = _apply_draft_quant(hidden_states, self._draft_lm_head_q)
            values, ids = _topk(logits, self.model.candidate_selector.top_k)
            ids = ids.to(torch.int64) + self.lm_head.shard_indices.org_vocab_start_index
            values = values.float()
            if self._cand_scale != 1.0:
                values = values * self._cand_scale
            if self._cand_pre_softcap is not None:
                values = torch.tanh(values / self._cand_pre_softcap) * self._cand_pre_softcap
            return ids, values
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k
        )


EntryClass = DFlash2Qwen3ForCausalLM
