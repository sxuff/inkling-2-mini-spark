#!/usr/bin/env python3
"""Long benchmark for Inkling 2 Mini behind an OpenAI-compatible server:
1. Single-stream decode (exact sxuff/inkling-2-mini-spark 3 prompts x reps, greedy, 160 tok, reasoning_effort=low + SHA-256 parity)
2. Single-stream long-decode (code/math/prose, greedy, 512 tok, reasoning_effort=low)
3. Cold prefill ladder (4K, 8K, 32K tokens, unique nonce, max_tokens=1)
4. Concurrent 4-stream and 8-stream coding/mixed batch (aggregate tok/s, per-stream tok/s, median/p90 TTFT)
5. Vision 2-image and 8-image exact color-order check
"""
import argparse, base64, concurrent.futures, hashlib, json, os, re, statistics, struct, sys, time, urllib.error, urllib.request, zlib

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

PROMPTS_SHORT = [
    ("code", "Write a self-contained Python 3 example of a bounded least-recently-used cache using OrderedDict, with typed methods and a full demonstration. Explain edge cases after the code."),
    ("prose", "Write a long scene about two coworkers restoring a 1980s synthesizer on a rainy afternoon. Use believable dialogue and concrete descriptions; write at least 350 words."),
    ("math", "Explain how to derive the variance of a sum of dependent random variables. Include a worked example for three variables and discuss both positive and negative covariance."),
]

CONC_PROMPTS = [
    "Write a complete, self-contained Rust implementation of a lock-free ring buffer (SPSC) with unit tests and memory ordering comments.",
    "Write a complete Python 3 async HTTP/1.1 connection pool with keep-alive, timeouts, per-host concurrency caps, and unit tests.",
    "Write a complete C++20 thread-safe LRU cache with O(1) get/put, TTL expiration, and a benchmark harness in main().",
    "Write a complete Go implementation of a Raft log replication state machine with term handling, AppendEntries, and table-driven tests.",
    "Write a complete TypeScript recursive-descent JSON5 parser with line/column error reporting and a comprehensive test suite.",
    "Write a complete Python 3 B+ tree implementation supporting insert, range scan, delete, and structural invariant verification.",
    "Write a complete C implementation of an arena allocator with alignment, scratch checkpoints, and stress tests.",
    "Write a complete SQL + Python SQLite migration runner with checksums, transactional rollback, and dry-run diffing.",
]

PALETTE = {"red": (222, 20, 18), "blue": (15, 70, 231), "green": (12, 151, 37), "yellow": (251, 230, 22),
           "purple": (134, 41, 173), "orange": (249, 125, 13), "white": (255, 255, 255), "black": (0, 0, 0)}

def stream_req(base, model, messages, *, max_tokens, temperature=0.0, ignore_eos=True, reasoning_effort="low", timeout=300):
    body = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = usage = finish = None
    reason, content = [], []
    chunk_toks = 0
    with OPENER.open(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b"data:"): continue
            line = line[5:].strip()
            if line == b"[DONE]": break
            j = json.loads(line)
            if j.get("usage"): usage = j["usage"]
            if not j.get("choices"): continue
            c = j["choices"][0]
            d = c.get("delta") or {}
            t = d.get("content")
            th = d.get("reasoning_content") or d.get("reasoning")
            if t or th:
                now = time.monotonic()
                if first is None: first = now
                last = now
                chunk_toks += 1
                if t: content.append(t)
                if th: reason.append(th)
            if c.get("finish_reason"): finish = c["finish_reason"]
    t1 = time.monotonic()
    n = (usage or {}).get("completion_tokens") or chunk_toks
    pt = (usage or {}).get("prompt_tokens") or 0
    dt = max(1e-6, (last or t1) - (first or t0))
    decode_tps = (n - 1) / dt if n > 1 else 0.0
    thoughts, text = "".join(reason), "".join(content)
    return {
        "ttft_s": (first - t0) if first else (t1 - t0),
        "wall_s": t1 - t0,
        "decode_tps": decode_tps,
        "completion_tokens": n,
        "prompt_tokens": pt,
        "finish": finish,
        "output_sha256": hashlib.sha256((thoughts + "\0" + text).encode()).hexdigest(),
        "text_head": (thoughts[:120] + " | " + text[:120]),
    }

def build_prefill_text(target_tokens, nonce):
    # ~4 chars per token of varied code-like filler so tokenizer doesn't collapse runs
    lines = [f"// Nonce: {nonce} — cold prefill measurement block"]
    for i in range(target_tokens // 14 + 40):
        lines.append(f"def compute_metric_{i:05d}(alpha={i%97}, beta={(i*7)%89}): return (alpha * {i+3} ^ beta) + {i*13 % 1009}")
    return "\n".join(lines) + "\n\nReply with just the integer 42."

def png_data_url(rgb):
    def chunk(k, v): return struct.pack(">I", len(v)) + k + v + struct.pack(">I", zlib.crc32(k + v))
    w = h = 96
    raw = (b"\0" + bytes(rgb) * w) * h
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}

def run_vision(base, model):
    names = list(PALETTE); items = [png_data_url(PALETTE[k]) for k in names]
    q = {"type": "text", "text": "List the dominant color of each image from all user turns in exactly the order received. Give only a comma-separated list of color names."}
    cases = [(2, [{"role": "user", "content": [*items[:2], q]}]),
             (8, [{"role": "user", "content": [{"type": "text", "text": "I am attaching the first four images. Wait for the next images."}, *items[:4]]},
                  {"role": "assistant", "content": "I will wait."},
                  {"role": "user", "content": [*items[4:], q]}])]
    rows = []
    for n, msgs in cases:
        body = {"model": model, "messages": msgs, "max_tokens": 512, "temperature": 0, "reasoning_effort": "low"}
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with OPENER.open(req, timeout=180) as r:
                j = json.load(r)
                text = j["choices"][0]["message"].get("content") or ""
                found = re.findall(r"\b(?:" + "|".join(names) + r")\b", text.lower())
                rows.append({"images": n, "http": r.status, "answer": text.strip(), "ok": found == names[:n]})
        except Exception as e:
            rows.append({"images": n, "error": str(e), "ok": False})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8020")
    ap.add_argument("--model", default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--skip-short", action="store_true")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--skip-conc", action="store_true")
    ap.add_argument("--no-ignore-eos", action="store_true")
    args = ap.parse_args()
    ign = not args.no_ignore_eos

    # Warmup / canary
    canary = stream_req(args.base, args.model, [{"role": "user", "content": "What is 17 times 24? Reply with only the integer."}],
                        max_tokens=64, temperature=0.0, ignore_eos=False)
    print(f"[canary] {canary['text_head']} ({canary['ttft_s']*1000:.0f} ms)", flush=True)

    res = {"base": args.base, "model": args.model, "canary": canary}

    if not args.skip_short:
        rows = []
        for rep in range(args.reps):
            for name, p in PROMPTS_SHORT:
                r = stream_req(args.base, args.model, [{"role": "user", "content": p}],
                               max_tokens=160, temperature=0.0, ignore_eos=ign)
                r["prompt"] = name; r["rep"] = rep
                rows.append(r)
                print(f"[short] {name:5s} rep={rep} {r['decode_tps']:6.2f} tok/s  ttft={r['ttft_s']*1000:5.0f} ms  n={r['completion_tokens']}  sha={r['output_sha256'][:10]}", flush=True)
        by_p = {k: statistics.median([x["decode_tps"] for x in rows if x["prompt"] == k]) for k, _ in PROMPTS_SHORT}
        res["short"] = {"rows": rows, "by_prompt": by_p, "median_tps": statistics.median(by_p.values())}
        print(f"[short SUMMARY] median={res['short']['median_tps']:.2f} | " + " ".join(f"{k}={v:.2f}" for k, v in by_p.items()), flush=True)

    if not args.skip_prefill:
        pf = []
        for target in (4096, 8192, 32768):
            nonce = f"{time.time_ns()}-{target}"
            txt = build_prefill_text(target, nonce)
            r = stream_req(args.base, args.model, [{"role": "user", "content": txt}],
                           max_tokens=1, temperature=0.0, ignore_eos=False, reasoning_effort="low")
            tps = r["prompt_tokens"] / max(1e-6, r["ttft_s"])
            row = {"target": target, "prompt_tokens": r["prompt_tokens"], "ttft_s": r["ttft_s"], "prefill_tps": tps}
            pf.append(row)
            print(f"[prefill] {target:5d} (actual {r['prompt_tokens']:5d}) -> {tps:7.1f} tok/s  ttft={r['ttft_s']:.3f}s", flush=True)
        res["prefill"] = pf

    if not args.skip_conc:
        conc = {}
        for n_streams in (4, 8):
            t0 = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_streams) as ex:
                futs = [ex.submit(stream_req, args.base, args.model,
                                  [{"role": "user", "content": CONC_PROMPTS[i % len(CONC_PROMPTS)]}],
                                  max_tokens=384, temperature=0.0, ignore_eos=ign)
                        for i in range(n_streams)]
                out_rows = [f.result() for f in futs]
            wall = time.monotonic() - t0
            tot_tok = sum(r["completion_tokens"] for r in out_rows)
            agg = tot_tok / wall
            mean_stream = statistics.mean(r["decode_tps"] for r in out_rows)
            ttfts = sorted(r["ttft_s"] for r in out_rows)
            conc[str(n_streams)] = {
                "wall_s": wall, "total_tokens": tot_tok, "agg_tps": agg,
                "mean_per_stream_tps": mean_stream,
                "ttft_median_s": statistics.median(ttfts),
                "ttft_p90_s": ttfts[min(len(ttfts)-1, int(0.9 * len(ttfts)))],
                "per_stream_tps": [round(r["decode_tps"], 2) for r in out_rows],
            }
            print(f"[conc C={n_streams}] agg={agg:6.2f} tok/s  wall={wall:5.2f}s  tot={tot_tok}  per_stream={mean_stream:5.2f}  ttft_med={conc[str(n_streams)]['ttft_median_s']*1000:.0f}ms  p90={conc[str(n_streams)]['ttft_p90_s']*1000:.0f}ms", flush=True)
        res["conc"] = conc

    if not args.skip_vision:
        vis = run_vision(args.base, args.model)
        res["vision"] = vis
        print(f"[vision] {[(v['images'], v['ok']) for v in vis]}", flush=True)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved -> {args.out}")

if __name__ == "__main__":
    main()
