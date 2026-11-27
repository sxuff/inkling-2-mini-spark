# Notices

This repository is licensed under the Apache License 2.0 (`LICENSE`).

## vLLM

`overlay/vllm/models/inkling2/nvidia/` and `patches/` contain files from [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0, copyright the vLLM contributors), taken from the image `vllm/vllm-openai@sha256:ecdf0ccc663d86332459f5a50a38c30861a5079bc3924aa08fdc657ef10dd648`:

- `dspark.py`: **modified**. Optional 8-bit copies of the draft-only weights (`INKLING2_DRAFT_QUANT`, `INKLING2_DRAFT_QUANT_LAYERS`).
- `ops/gemv.py`: **modified**. 16×512 tiles for wide projections (`INKLING2_GEMV_TILE`).
- `ops/gemv_q8.py`: **new file**, built on vLLM's Triton utilities.

Each file carries a notice of its change. `patches/inkling2-gb10-draft-int8.patch` is the full diff against the image's originals.

`overlay/dflash2/` holds the files for the `dflash2` profile, from the same image:

- `vllm/model_executor/models/qwen3_dflash2.py`: **modified**. Draft-only INT8/INT4/NVFP4 weight copies (`DFLASH2_HEAD_QUANT`, `DFLASH2_BODY_QUANT`), fused convolution + residual + norm (`DFLASH2_FUSED_OPS`), selector softcap placement, embedding norm.
- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`: **modified**. Skips draft context K/V for prefill positions outside the drafter's sliding window (`DFLASH2_SKIP_EXPIRED_CTX`).
- `vllm/v1/worker/gpu/spec_decode/dflash/utils.py`: **modified**. Shares the embedding norm, builds the draft-only copies after loading.
- `vllm/models/inkling2/nvidia/ops/gemv.py`: **modified**. Tile choice by output width.
- `vllm/models/inkling2/nvidia/ops/gemv_q4.py`: **new file**, built on vLLM's Triton utilities and `gemv_q8.py`.

`patches/inkling2-gb10-dflash2.patch` is the full diff against the image's originals.

The Inkling2 model implementation, the DSpark and DFlash speculators and the Humming expert backend these files plug into are vLLM's work.

## Inkling 2 Mini

The model is [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. Its weights are not redistributed here. `scripts/download.py` fetches the pinned revision from Hugging Face and verifies every file.

## DFlash 2 drafter

The `dflash2` profile drafts with [incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) by Inco AI, Apache-2.0. Its weights are not redistributed here. `scripts/download.py --dflash2` fetches the pinned revision from Hugging Face and verifies every file.
