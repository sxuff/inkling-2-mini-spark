# Startup A/B (2026-11-22)

Run with [0xSero](https://github.com/0xSero)'s [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill) and its `bench.py`, against the q8 profile on one GX10. Each candidate sits between control runs of the unchanged launch. `ledger.jsonl` has one line per run: startup, c1 decode (median of 6 sampled prompts, 512 tokens), c4 aggregate, prefill at 3.3K and 25.6K tokens, TTFT and memory peak.

| Run | Change | Startup (s) | c1 tok/s | Prefill 25.6K (tok/s) | Verdict |
|---|---|---:|---:|---:|---|
| ctl-a, ctl-b, ctl-f, ctl-g, ctl-i | none | 127–155 | 43.86–43.99 | 3,819–3,858 | control |
| eager, eager2 | `--safetensors-load-strategy eager` | 42.2, 45.2 | 43.91, 43.80 | 3,812, 3,813 | **kept** |
| mnbt8k | `--max-num-batched-tokens 8192` | 155.3 | 43.86 | 3,932 | within noise |
| mnbt16k | `--max-num-batched-tokens 16384` | 162.3 | 43.83 | 3,278 | rejected |
| async | `--async-scheduling` | 151.3 | 43.80 | 3,811 | no change |
| k5 | 5 draft tokens | — | — | — | not runnable: DSpark needs exactly 7 |

`eager2` also passed a 102,426-token needle test (3/3 codes, 66.2 tok/s decode at that length). c1 here is lower than the main results table because this prompt panel includes long-form prose, which the drafter predicts less well; per prompt it ranged from 37 to 66 tok/s. The c4 figure is one wall-clock sample per run and moved between 89 and 98 tok/s on identical launches, so it is not used for decisions. `ctl-h` is void (a container-name race in the harness).
