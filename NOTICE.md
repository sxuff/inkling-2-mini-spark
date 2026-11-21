# Notices

This repository is licensed under the Apache License 2.0 (`LICENSE`).

## vLLM

`overlay/vllm/models/inkling2/nvidia/` and `patches/` contain files from [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0, copyright the vLLM contributors), taken from the image `vllm/vllm-openai@sha256:ecdf0ccc663d86332459f5a50a38c30861a5079bc3924aa08fdc657ef10dd648`:

- `dspark.py`: **modified**. Optional 8-bit copies of the draft-only weights (`INKLING2_DRAFT_QUANT`, `INKLING2_DRAFT_QUANT_LAYERS`).
- `ops/gemv.py`: **modified**. 16×512 tiles for wide projections (`INKLING2_GEMV_TILE`).
- `ops/gemv_q8.py`: **new file**, built on vLLM's Triton utilities.

Each file carries a notice of its change. `patches/inkling2-gb10-draft-int8.patch` is the full diff against the image's originals.

The Inkling2 model implementation, the DSpark speculator and the Humming expert backend these files plug into are vLLM's work.

## Inkling 2 Mini

The model is [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. Its weights are not redistributed here. `scripts/download.py` fetches the pinned revision from Hugging Face and verifies every file.
