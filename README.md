# Inkling 2 Mini on a single DGX Spark

![Inkling 2 Mini on one GB10: 66.7 tok/s median decode, same outputs as stock vLLM](assets/card.png)

This recipe serves [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single NVIDIA GB10 (DGX Spark, ASUS GX10) with vLLM and the model's bundled DSpark speculative decoder. It adds three small kernel changes on top of the pinned vLLM image. You get an OpenAI-compatible endpoint with a 262K-token context, image input, reasoning and tool calls.

- **66.7 tok/s** median single-stream decode on the live service, up from **60.5** with the same image, checkpoint and arguments (+10%).
- **Same text:** all 9 benchmark outputs are byte-identical to stock vLLM. The changes only make the draft model cheaper, and the target model still verifies every token.
- **80 tok/s** on a 1,500-token code answer, first token in 190 ms ([real-time capture](assets/demo.mp4)).
- **262,144-token context:** a 262,016-token prompt was served, and a 262,145-token prompt was rejected.
- **Native vision:** 2- and 8-image color checks passed, including images spread over two user turns.

## Quick start

You need one GB10 with Docker and the NVIDIA container runtime, about 50 GB of free disk and about 75 GB of free unified memory. The server takes 61% of the GPU memory, which leaves room for a desktop session.

```bash
git clone https://github.com/sxuff/inkling-2-mini-gb10
cd inkling-2-mini-gb10
python3 scripts/download.py   # pinned checkpoint (20.2 GB) into ~/models, every file verified
./scripts/serve.sh            # q8 profile on 127.0.0.1:8020
```

The image is about 27 GB, and `serve.sh` pulls it on first use. With the image already pulled and an empty cache, the first start took 230 s on the test machine. Later starts reuse `~/.cache/inkling2-mini-vllm` and take about 45 s, because `serve.sh` reads the checkpoint in one pass instead of memory-mapping it (see Startup below).

```bash
curl -s http://127.0.0.1:8020/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "thinkingmachines/Inkling-2-Mini",
  "messages": [{"role": "user", "content": "What is 17 times 24?"}],
  "reasoning_effort": "low"
}'
```

- **Reasoning** comes back in a separate `reasoning_content` field. `reasoning_effort` sets how much the model thinks.
- **Tool calls** use vLLM's `inkling` tool parser, which `serve.sh` already enables.
- **Images** go in as normal OpenAI `image_url` parts. Video input is off.
- **Profiles:** `serve.sh humming` runs the same server without this recipe's kernels, and `serve.sh stock` also drops the Humming expert backend. Use them for A/B runs.
- **Overrides:** `serve.sh` takes `MODEL_DIR`, `PORT`, `GPU_UTIL`, `MAX_LEN`, `MAX_SEQS`, `CACHE_DIR` and `CONTAINER` from the environment.

## Results

All numbers are single-stream decode after the first token. Each run sends 9 serial requests (3 prompts × 3 repeats) with greedy sampling and exactly 160 generated tokens, reasoning included. Decode tok/s = (tokens − 1) / (last − first streamed token). The arms ran back to back on one boot with the same image, checkpoint and vLLM arguments.

| Profile | Median tok/s | Code | Prose | Math | Tokens per step | Outputs = stock |
|---|---:|---:|---:|---:|---:|---:|
| Stock vLLM, Humming experts (`serve.sh humming`) | 60.51 | 62.55 | 41.66 | 60.51 | 3.10 | 9/9 |
| + wider BF16 GEMV tiles | 61.04 | 63.08 | 41.96 | 61.04 | 3.10 | 9/9 |
| + INT8 draft LM head and Markov fold | 65.20 | 71.91 | 44.55 | 65.20 | 3.16 | 9/9 |
| **+ INT8 draft layers (`serve.sh q8`)** | **66.43** | **73.06** | **45.44** | **66.43** | 3.16 | 9/9 |
| Stock again, after the A/B | 60.50 | 62.41 | 41.70 | 60.50 | 3.10 | 9/9 |
| **Live service after promotion** | **66.72** | **73.29** | **45.55** | **66.72** | | 9/9 |

- **First token:** 164 ms median with the q8 profile, against 171 ms on stock.
- **Tokens per step** is the mean number of tokens each speculative step commits (1 + accepted drafts). The INT8 drafter accepts slightly more, not less.
- **Memory:** the INT8 copies add 0.74 GiB, so the KV cache shrinks from 717,758 to 703,530 tokens. That is still 2.68 full 262K requests.
- **Clean-clone check:** `download.py` verification, then `serve.sh q8` from a fresh copy of this repository with an empty cache: 66.62 tok/s, all 9 hashes equal to stock, both vision checks passed (`results/recipe-test/`).
- **Startup:** `--safetensors-load-strategy eager` cuts a warm start from 127–155 s to 42–45 s (two runs, five interleaved controls). Weight loading drops from 100 s to 14 s; decode, prefill, memory and a 102K-token needle test (3/3 found) are unchanged. On the live service the restart took 47 s. This came from a run of [0xSero](https://github.com/0xSero)'s [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill); the same run ruled out larger prefill chunks (8K within noise, 16K 14% slower on a 25K-token prompt) and async scheduling (no change). The ledger is in `results/startup/`.
- **Raw data:** `results/ab/` has every request, with timings, output text and hashes. `results/live/` has the promotion checks: bench, vision, and context limit.

## How it works

### Where a decode step goes

A torch-profiler trace of the stock service (`results/profile/`) shows the GPU busy 99.4% of each decode step, so the CPU and kernel launches are not the bottleneck. Each step verifies 8 tokens (1 + 7 drafts) and then drafts the next 7, about 50 ms in total.

| Kernel group | Share of GPU time | Notes |
|---|---:|---|
| Humming NVFP4 expert GEMMs | 43% | ~250 GB/s, at the memory roofline |
| BF16 GEMVs | 34% | attention projections, target LM head, and the draft's LM head and layers |
| cuBLAS (draft Markov fold, loop controllers) | 9% | the Markov fold reads 103 MB, 7 times per step |
| MXFP8 expert GEMMs (layers 20–23) | 8% | |
| Everything else | 6% | attention, routing, norms |

The quantize and combine kernels look expensive in a naive trace. They are not: they launch early (PDL) and wait on the GEMM before them. `tools/kernel_time.py` bills each instant to the earliest-started kernel, which gives the table above.

The target's weights already stream at close to GB10's 273 GB/s. The draft, though, re-reads a full BF16 copy of the 0.9 GB LM head on every step, and its recurrent Markov head re-reads a 103 MB projection 7 times. That's about 7 ms of every 50 ms step spent on weights only the drafter uses.

### The changes

Three vLLM modules are mounted read-only over the pinned image (`overlay/`, also as `patches/inkling2-gb10-draft-int8.patch`). `serve.sh` checks each file's SHA-256 before launch.

1. **`ops/gemv_q8.py` (new):** a Triton weight-only 8-bit GEMV with an fp32 scale per output row. The 8-bit values convert exactly to BF16, so each tile runs the same BF16 tensor-core dot as the stock GEMV with half the weight bytes.
2. **`dspark.py`:** with `INKLING2_DRAFT_QUANT=int8`, the drafter's base logits and the Markov fold run from INT8 copies made at load time. `INKLING2_DRAFT_QUANT_LAYERS=1` adds the three draft layers. Prefill-sized batches keep BF16.
3. **`ops/gemv.py`:** BF16 GEMVs with N ≥ 4096 use 16×512 tiles instead of 32×256. Each output column keeps the same K-ordered tensor-core chain, so results are bit-identical. `INKLING2_GEMV_TILE=0` restores the stock tiles.

Measured on the GB10 (`results/kernels/`):

| Projection | Stock | This recipe |
|---|---:|---:|
| Draft LM head (201,024 × 2,304), 7–8 rows | 3,613 µs BF16 | 1,780 µs INT8 |
| Markov fold, 1 row (201,024 × 256), 7× per step | 404 µs BF16 (cuBLAS) | 205 µs INT8 |
| Target LM head, 8 rows | 3,613 µs | 3,529 µs (16×512 tiles, bit-identical) |

**Why INT8 and not FP8:** on the real LM head, INT8 with per-row scales has 0.8–0.9% relative error and a softcapped-softmax KL of 4e-5. FP8 (e4m3) has 2.4–2.7% and 3.5e-4, at the same speed.

**Why the text doesn't change:** speculative decoding accepts or rejects each draft token with the target model's own BF16/NVFP4/MXFP8 weights. With greedy decoding, the output is the target's argmax whatever the draft proposes. With sampling, vLLM stores the logits each draft token was sampled from, here the INT8 ones, and rejection sampling against them stays exact. A cheaper draft can only change how many tokens each step accepts.

## Configuration

| Component | Pin |
|---|---|
| Model | `thinkingmachines/Inkling-2-Mini` @ `ed40e7da625f594fc9c4990f5f66be24873c22d7`: native NVFP4 experts (layers 0–19), MXFP8 experts (20–23), BF16 everything else, bundled DSpark draft |
| Runtime | `vllm/vllm-openai` @ `sha256:ecdf0ccc663d86332459f5a50a38c30861a5079bc3924aa08fdc657ef10dd648` (ARM64; vLLM 0.1.dev25855+g7cb5dae22, PyTorch 2.14, Triton 3.8) |
| Speculation | `dspark`, 7 tokens (fixed by the checkpoint's block size), probabilistic draft sampling |
| Serving | `--moe-backend humming --safetensors-load-strategy eager --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.61` |
| Recipe env | `INKLING2_DRAFT_QUANT=int8`, `INKLING2_DRAFT_QUANT_LAYERS=1` |

## Reproduce

With the server running:

```bash
python3 scripts/bench.py my-run.json --compare results/ab/stock.json
```

The script prints the median and per-prompt tok/s, tokens per step, and how many outputs match the stock run hash for hash. It also runs the 2- and 8-image vision check. For an A/B, start `serve.sh humming` and `serve.sh q8` in turn and run the bench against each.

The kernel labs run inside the image with this repository mounted at `/w` and the checkpoint at `/model`:

```bash
docker run --rm --gpus all -v "$PWD:/w" -v "$HOME/models/Inkling-2-Mini-ed40e7d:/model:ro" \
  --entrypoint python3 vllm/vllm-openai@sha256:ecdf0ccc663d86332459f5a50a38c30861a5079bc3924aa08fdc657ef10dd648 \
  /w/tools/q8_accuracy.py   # or gemv_tiles.py, fp8_gemv_lab.py
```

### Record a demo

`demo/` holds the page and capture script behind `assets/demo.mp4`. The capture is real time: Chromium's DevTools screencast records the page while it streams from the server, and ffmpeg encodes the frames at their real timestamps. It needs Python Playwright, Chromium and ffmpeg.

```bash
python3 demo/record.py demo.mp4 "max=1500"   # streams from 127.0.0.1:8020 (UPSTREAM_PORT to change)
```

## Limits

- **What was tested:** single-stream greedy text, the vision check, and the context limit. Sampled requests and concurrent requests were not benchmarked with the q8 profile. They should keep their output distribution, but the speed-up there is unmeasured.
- **Draft length is fixed:** Inkling 2's DSpark requires exactly 7 speculative tokens, and vLLM's adaptive verification is off for this model.
- **What's left is lossy:** the target's expert and attention weights already run near the memory roofline. Bigger wins would need lossy changes to the target model, such as 8-bit attention projections, which would change outputs.

## License and credits

Apache-2.0 (`LICENSE`). The files in `overlay/` are modified from, or built on, vLLM (Apache-2.0) and carry change notices. The Inkling 2 Mini weights are Apache-2.0 and are not included here. See `NOTICE.md`.

The startup change was found with [0xSero](https://github.com/0xSero)'s [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill) (MIT): its ceiling-first, control-interleaved A/B method, bench harness and memory watchdog ran unchanged on the GB10.
