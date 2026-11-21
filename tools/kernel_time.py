# Exclusive GPU time per kernel from a torch-profiler trace: each instant goes to the earliest-started
# running kernel, so PDL (programmatic dependent launch) waiters are not billed for their predecessor.
# usage: python3 kernel_time.py TRACE_DIR
# exclusive attribution: at each instant, time goes to the active kernel that started earliest (PDL waiters get nothing while predecessor runs)
import json, gzip, sys, glob, collections, re, os, heapq
f = sorted(glob.glob(sys.argv[1] + '/**/*.pt.trace.json*', recursive=True), key=lambda p: -os.path.getsize(p))[0]
tr = json.load(gzip.open(f)); ev = tr['traceEvents']
k = sorted([e for e in ev if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')], key=lambda e: e['ts'])
def norm(n):
    n = re.sub(r'<.*', '', n); return n[:70]
# boundaries
pts = sorted({e['ts'] for e in k} | {e['ts'] + e['dur'] for e in k})
starts = collections.defaultdict(list)
for i, e in enumerate(k): starts[e['ts']].append(i)
active = []  # heap by start
excl = collections.defaultdict(float); overlap_wait = collections.defaultdict(float)
idle = 0.0
ends = {}
for i, e in enumerate(k): ends.setdefault(e['ts'] + e['dur'], []).append(i)
alive = set()
for a, b in zip(pts, pts[1:]):
    for i in starts.get(a, []): heapq.heappush(active, (k[i]['ts'], i)); alive.add(i)
    for i in ends.get(a, []): alive.discard(i)
    while active and active[0][1] not in alive: heapq.heappop(active)
    if not active: idle += b - a; continue
    first = active[0][1]
    excl[norm(k[first]['name'])] += b - a
    for _, i in active:
        if i != first and i in alive: overlap_wait[norm(k[i]['name'])] += b - a
tot = sum(excl.values())
print(f'total attributed {tot/1e3:.1f} ms, idle {idle/1e3:.1f} ms')
cnt = collections.Counter(norm(e['name']) for e in k)
print(f'{"kernel":70s} {"excl_ms":>8s} {"%":>6s} {"n":>6s} {"us/call":>8s} {"waitms":>8s}')
for n, d in sorted(excl.items(), key=lambda x: -x[1])[:30]:
    print(f'{n:70s} {d/1e3:8.2f} {100*d/tot:6.2f} {cnt[n]:6d} {d/cnt[n]:8.1f} {overlap_wait[n]/1e3:8.1f}')
