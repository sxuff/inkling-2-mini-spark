#!/usr/bin/env python3
"""Record real per-token timestamps from a vLLM OpenAI server for the speed video.

Runs ON the GX10 (localhost) so the timestamps carry no network jitter.

  python3 capture.py --port 8020 --mode single --out single.json
  python3 capture.py --port 8020 --mode multi  --out multi.json

Each stream is saved as a list of events [t_seconds_since_request, kind, text, completion_tokens]
where kind is "r" (reasoning) or "c" (answer content). completion_tokens comes from vLLM's
continuous usage stats, so tok/s in the video is the server's own token count, not an estimate.
"""
import argparse, concurrent.futures, json, time, urllib.request

MODEL = "thinkingmachines/Inkling-2-Mini"

SINGLE = {
    "label": "Code",
    "prompt": "Write a production-quality Python implementation of an LRU cache with TTL expiry, "
              "thread safety, and hit/miss stats. Include type hints, docstrings and a short usage example.",
}

MULTI = [
    {"label": "Python", "prompt": "Write a Python async web crawler with a concurrency limit, retries with backoff, and robots.txt support."},
    {"label": "Rust", "prompt": "Implement a lock-free single-producer single-consumer ring buffer in Rust with tests."},
    {"label": "SQL", "prompt": "Design a Postgres schema for a multi-tenant SaaS billing system, then write the 3 most important queries."},
    {"label": "Math", "prompt": "Prove that there are infinitely many primes of the form 4k+3, step by step."},
    {"label": "TypeScript", "prompt": "Write a React hook in TypeScript for infinite scroll with cancellation and error states."},
    {"label": "Prose", "prompt": "Write a vivid short story opening about a lighthouse keeper who receives a letter from the future."},
    {"label": "Explain", "prompt": "Explain how speculative decoding works to a smart engineer who has never heard of it."},
    {"label": "Bash", "prompt": "Write a robust bash script that backs up a directory to S3 with rotation, logging and a dry-run flag."},
]


def stream(port, prompt, max_tokens, temperature, effort, t_start_barrier=None):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": effort,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    if t_start_barrier is not None:
        t_start_barrier.wait()
    t0 = time.perf_counter()
    events, usage = [], None
    with op.open(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            t = time.perf_counter() - t0
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                ntok = (obj.get("usage") or {}).get("completion_tokens")
                for kind, key in (("r", "reasoning_content"), ("r", "reasoning"), ("c", "content")):
                    if d.get(key):
                        events.append([round(t, 5), kind, d[key], ntok])
    return {"t_total": time.perf_counter() - t0, "usage": usage, "events": events}


def summarize(name, res):
    ev = res["events"]
    toks = (res["usage"] or {}).get("completion_tokens") or 0
    ttft = ev[0][0] if ev else 0
    decode = res["t_total"] - ttft
    print(f"{name:12s} tokens={toks:5d} ttft={ttft:.3f}s decode={decode:.2f}s "
          f"decode_tok/s={toks / decode if decode else 0:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--mode", choices=["single", "multi"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--effort", default="low")
    a = ap.parse_args()

    meta = {"model": MODEL, "port": a.port, "temperature": a.temperature, "effort": a.effort,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if a.mode == "single":
        mt = a.max_tokens or 1024
        res = stream(a.port, SINGLE["prompt"], mt, a.temperature, a.effort)
        summarize(SINGLE["label"], res)
        out = {**meta, "mode": "single", "max_tokens": mt,
               "streams": [{**SINGLE, **res}]}
    else:
        import threading
        mt = a.max_tokens or 512
        barrier = threading.Barrier(len(MULTI))
        with concurrent.futures.ThreadPoolExecutor(len(MULTI)) as ex:
            futs = [ex.submit(stream, a.port, m["prompt"], mt, a.temperature, a.effort, barrier) for m in MULTI]
            results = [f.result() for f in futs]
        streams = [{**m, **r} for m, r in zip(MULTI, results)]
        for s in streams:
            summarize(s["label"], s)
        total = sum((s["usage"] or {}).get("completion_tokens", 0) for s in streams)
        wall = max(s["t_total"] for s in streams)
        print(f"AGGREGATE    tokens={total} wall={wall:.2f}s tok/s={total / wall:.2f}")
        out = {**meta, "mode": "multi", "max_tokens": mt, "streams": streams}
    with open(a.out, "w") as f:
        json.dump(out, f)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
