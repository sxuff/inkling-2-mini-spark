#!/usr/bin/env python3
"""Single-stream decode benchmark for the running server (the protocol behind results/).

usage: python3 scripts/bench.py OUT.json [--port 8020] [--compare results/ab/stock.json] [--no-vision]

Protocol: canary (17 x 24), one warmup, then 3 prompts x 3 repeats, serial, greedy (temperature 0), exactly
160 generated tokens (ignore_eos), reasoning_effort low. Decode tok/s = (completion_tokens - 1) /
(last - first non-empty streamed delta), reasoning tokens included, TTFT excluded. Each row stores a SHA-256
of reasoning + NUL + answer, so runs can be compared token for token. A vision check sends 2 and 8
solid-color images (8 across two user turns) and expects the colors in order. Stdlib only.
"""
import argparse, base64, hashlib, json, re, statistics, struct, time, urllib.error, urllib.request, zlib

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--port", type=int, default=8020)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--compare", help="earlier bench JSON to compare output hashes against")
ap.add_argument("--no-vision", action="store_true")
ns = ap.parse_args()
BASE = f"http://{ns.host}:{ns.port}"
MODEL = "thinkingmachines/Inkling-2-Mini"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PROMPTS = [
    ("code", "Write a self-contained Python 3 example of a bounded least-recently-used cache using OrderedDict, with typed methods and a full demonstration. Explain edge cases after the code."),
    ("prose", "Write a long scene about two coworkers restoring a 1980s synthesizer on a rainy afternoon. Use believable dialogue and concrete descriptions; write at least 350 words."),
    ("math", "Explain how to derive the variance of a sum of dependent random variables. Include a worked example for three variables and discuss both positive and negative covariance."),
]


def request(path, body=None, timeout=300):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    return OPENER.open(req, timeout=timeout)


def post(path, body, timeout=300):
    try:
        with request(path, body, timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read(4096).decode(errors="replace")


def chat(text, **kw):
    return {"model": MODEL, "messages": [{"role": "user", "content": text}], **kw}


def stream(body):
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    start = time.monotonic(); first = last = usage = finish = None; reason, content = [], []
    with request("/v1/chat/completions", body) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            line = line[5:].strip()
            if line == b"[DONE]":
                break
            j = json.loads(line)
            usage = j.get("usage") or usage
            if not j.get("choices"):
                continue
            c = j["choices"][0]; d = c.get("delta", {})
            t = d.get("content"); th = d.get("reasoning_content") or d.get("reasoning")
            if t or th:
                now = time.monotonic(); first = first or now; last = now
                if t: content.append(t)
                if th: reason.append(th)
            finish = c.get("finish_reason") or finish
    n = usage["completion_tokens"]; thoughts, text = "".join(reason), "".join(content)
    return {"ttft_s": first - start, "decode_tps": (n - 1) / (last - first), "usage": usage, "finish": finish,
            "text": text, "output_sha256": hashlib.sha256((thoughts + "\0" + text).encode()).hexdigest()}


def spec_counters():
    with request("/metrics", timeout=10) as r:
        txt = r.read().decode()
    out = {}
    for name in ("num_drafts_total", "num_draft_tokens_total", "num_accepted_tokens_total"):
        m = re.search(rf"^vllm:spec_decode_{name}\{{[^}}]*\}} ([0-9.e+]+)$", txt, re.M)
        out[name] = float(m.group(1)) if m else 0.0
    return out


def png(rgb):
    def chunk(k, v): return struct.pack(">I", len(v)) + k + v + struct.pack(">I", zlib.crc32(k + v))
    w = h = 96
    raw = (b"\0" + bytes(rgb) * w) * h
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


PALETTE = {"red": (222, 20, 18), "blue": (15, 70, 231), "green": (12, 151, 37), "yellow": (251, 230, 22),
           "purple": (134, 41, 173), "orange": (249, 125, 13), "white": (255, 255, 255), "black": (0, 0, 0)}


def vision():
    names = list(PALETTE); items = [png(PALETTE[k]) for k in names]
    q = {"type": "text", "text": "List the dominant color of each image from all user turns in exactly the order received. Give only a comma-separated list of color names."}
    cases = [(2, [{"role": "user", "content": [*items[:2], q]}]),
             (8, [{"role": "user", "content": [{"type": "text", "text": "I am attaching the first four images. Wait for the next images."}, *items[:4]]},
                  {"role": "assistant", "content": "I will wait."},
                  {"role": "user", "content": [*items[4:], q]}])]
    rows = []
    for n, msgs in cases:
        code, j = post("/v1/chat/completions", {"model": MODEL, "messages": msgs, "max_tokens": 512, "temperature": 0, "reasoning_effort": "low"})
        text = (j["choices"][0]["message"].get("content") or "") if code == 200 else ""
        found = re.findall(r"\b(?:" + "|".join(names) + r")\b", text.lower())
        rows.append({"images": n, "http": code, "answer": text, "matches_in_order": found == names[:n]})
    return rows


for _ in range(600):
    try:
        with request("/health", timeout=5) as r:
            if r.status == 200: break
    except OSError:
        time.sleep(2)
out = {"protocol": __doc__.split("\n\n")[1].replace("\n", " ")}
code, j = post("/v1/chat/completions", chat("What is 17 times 24? Give the number and nothing else.", temperature=0, max_tokens=512, reasoning_effort="low"))
out["canary"] = {"http": code, "answer": j["choices"][0]["message"]["content"] if code == 200 else j}
body = lambda p: chat(p, temperature=0, top_p=1, max_tokens=160, ignore_eos=True, reasoning_effort="low")
stream(body("Explain why fresh snow looks white."))
before = spec_counters(); rows = []
for rep in range(ns.reps):
    for name, prompt in PROMPTS:
        row = stream(body(prompt)); row.update(case=name, repeat=rep); rows.append(row)
        print(f"{name:6s} rep {rep}  {row['decode_tps']:6.2f} tok/s  ttft {row['ttft_s'] * 1000:4.0f} ms", flush=True)
after = spec_counters()
out["rows"] = rows
out["median_decode_tps"] = statistics.median(r["decode_tps"] for r in rows)
out["median_ttft_s"] = statistics.median(r["ttft_s"] for r in rows)
out["case_medians"] = {n: statistics.median(r["decode_tps"] for r in rows if r["case"] == n) for n, _ in PROMPTS}
d = {k: after[k] - before[k] for k in after}
if d["num_drafts_total"]:
    out["mean_acceptance_length"] = 1 + d["num_accepted_tokens_total"] / d["num_drafts_total"]
if ns.compare:
    ref = json.load(open(ns.compare))["rows"]
    out["matched_hashes"] = sum(a["output_sha256"] == b["output_sha256"] for a, b in zip(rows, ref))
if not ns.no_vision:
    out["vision"] = vision()
json.dump(out, open(ns.out, "w"), indent=1)
print(f"median {out['median_decode_tps']:.2f} tok/s | " + " ".join(f"{k} {v:.2f}" for k, v in out["case_medians"].items())
      + f" | accept {out.get('mean_acceptance_length', 0):.3f} | canary {out['canary']['answer']!r}"
      + (f" | hashes {out['matched_hashes']}/{len(rows)}" if ns.compare else "")
      + ("" if ns.no_vision else " | vision " + " ".join(f"{v['images']}:{'ok' if v['matches_in_order'] else 'FAIL'}" for v in out["vision"])))
