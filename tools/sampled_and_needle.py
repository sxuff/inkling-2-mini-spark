#!/usr/bin/env python3
import argparse, json, pathlib, time, urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def stream_post(base, body, timeout=600):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    first = last = usage = finish = None
    reason, content = [], []
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
                if t: content.append(t)
                if th: reason.append(th)
            if c.get("finish_reason"): finish = c["finish_reason"]
    t1 = time.monotonic()
    n = (usage or {}).get("completion_tokens", 0)
    pt = (usage or {}).get("prompt_tokens", 0)
    dt = max(1e-6, (last or t1) - (first or t0))
    return {
        "prompt_tokens": pt, "completion_tokens": n,
        "ttft_s": (first - t0) if first else (t1 - t0),
        "decode_tps": (n - 1) / dt if n > 1 else 0.0,
        "content_head": "".join(content)[:200],
    }

def meminfo():
    d = {}
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        d[k] = int(v.strip().split()[0]) * 1024
    return {
        "mem_avail_gib": round(d["MemAvailable"] / (1 << 30), 2),
        "swap_used_mib": round((d["SwapTotal"] - d["SwapFree"]) / (1 << 20), 2),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8020")
    ap.add_argument("--model", default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-needle", action="store_true")
    args = ap.parse_args()

    long_prompt = (
        "Write a complete, production-quality Python 3 async HTTP/1.1 connection pool "
        "with keep-alive, per-host concurrency limits, exponential backoff retries, "
        "connection health checks, structured metrics, and a thorough unittest suite."
    )
    prose_prompt = (
        "Write a long scene about two coworkers restoring a 1980s synthesizer on a rainy afternoon. "
        "Use believable dialogue and concrete descriptions; write at least 350 words."
    )
    out = {"mem_before": meminfo()}
    # Warmup
    stream_post(args.base, {"model": args.model, "messages": [{"role": "user", "content": "What is 17*24?"}], "max_tokens": 32, "temperature": 0.0, "reasoning_effort": "low"})

    for label, prompt, temp, max_t in [
        ("code_T0_1024", long_prompt, 0.0, 1024),
        ("code_T07_1024", long_prompt, 0.7, 1024),
        ("code_T10_1024", long_prompt, 1.0, 1024),
        ("prose_T07_512", prose_prompt, 0.7, 512),
    ]:
        body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_t, "temperature": temp, "ignore_eos": True, "reasoning_effort": "low"}
        if temp > 0:
            body["top_p"] = 0.95
            body["seed"] = 42
        r = stream_post(args.base, body)
        out[label] = r
        print(f"  {label:15s}: {r['decode_tps']:6.2f} tok/s  ttft={r['ttft_s']*1000:4.0f}ms  toks={r['completion_tokens']}", flush=True)

    if args.with_needle:
        needle = "The secret sapphire verification code for Project Orion is: ORION-7492-DELTA."
        line_tpl = "Log {:06d}: alpha={:.2f} pressure={:.1f} status=OK crc=0x{:04x}\n"
        # ~24 tokens/line -> 8400 lines ~ 200,000 tokens (let's use 8000 lines ~ 195K tokens, safely < 262144!)
        half = 4000
        parts = [line_tpl.format(i, (i % 97) * 0.37, 100.0 + (i % 41), (i * 2654435761) & 0xFFFF) for i in range(half)]
        parts.append(f"\nIMPORTANT ARCHIVE NOTE: {needle}\n\n")
        parts.extend(line_tpl.format(i, (i % 97) * 0.37, 100.0 + (i % 41), (i * 2654435761) & 0xFFFF) for i in range(half, 2 * half))
        haystack = "".join(parts) + "\nQuestion: What is the secret sapphire verification code for Project Orion? Reply with ONLY the code."
        r_n = stream_post(args.base, {
            "model": args.model,
            "messages": [{"role": "user", "content": haystack}],
            "max_tokens": 96, "temperature": 0.0, "reasoning_effort": "low",
        }, timeout=600)
        ok = "ORION-7492-DELTA" in (r_n["content_head"] or "")
        pf_tps = r_n["prompt_tokens"] / max(1e-6, r_n["ttft_s"])
        out["needle_200k"] = {
            "ok": ok, "prompt_tokens": r_n["prompt_tokens"],
            "ttft_s": round(r_n["ttft_s"], 2), "prefill_tps": round(pf_tps, 1),
            "decode_tps": round(r_n["decode_tps"], 2), "answer": r_n["content_head"].strip(),
        }
        print(f"  200K NEEDLE: ok={ok} prompt_toks={r_n['prompt_tokens']} ttft={r_n['ttft_s']:.2f}s ({pf_tps:.1f} tok/s) ans={r_n['content_head'].strip()!r}", flush=True)

    out["mem_after"] = meminfo()
    print(f"  memory: before={out['mem_before']} after={out['mem_after']}", flush=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

if __name__ == "__main__":
    main()
