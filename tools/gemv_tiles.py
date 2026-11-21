# Bit-exactness and bandwidth of BF16 GEMV tile shapes vs the stock Inkling2 GEMV (run inside the image).
"""GEMV lab: stream many distinct weight matrices back-to-back in one CUDA graph (like the model) and compare variants."""
import torch, statistics, sys, itertools, json
import triton, triton.language as tl
from vllm.models.inkling2.nvidia.ops.gemv import sm12x_bf16_gemv
dev = 'cuda'; torch.manual_seed(0)

@triton.jit
def gemv_v(X, W, Y, M, N, K, SX, SW, SY, BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr, SWZ: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(W + cols[None, :].to(tl.int64) * SW + ks[:, None], (cols[None, :] < N) & (ks[:, None] < K), other=0)
        x = tl.load(X + rows[:, None] * SX + ks[None, :], (rows[:, None] < M) & (ks[None, :] < K), other=0)
        acc += tl.dot(x, w)
    tl.store(Y + rows[:, None] * SY + cols[None, :], acc.to(Y.dtype.element_ty), (rows[:, None] < M) & (cols[None, :] < N))

def run_v(cfg):
    BN, BK, nw, ns = cfg
    def f(x, w, out):
        m, k = x.shape; n = w.shape[0]
        gemv_v[(triton.cdiv(n, BN),)](x, w, out, m, n, k, x.stride(0), w.stride(0), out.stride(0), BN=BN, BK=BK, MP=16, SWZ=0, num_warps=nw, num_stages=ns)
        return out
    return f
def stock(x, w, out): return sm12x_bf16_gemv(x, w, out_dtype=out.dtype, out=out)

def timeit(fn, xs, ws, outs, reps=15):
    s = torch.cuda.Stream(); g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        for x, w, o in zip(xs, ws, outs): fn(x, w, o)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for x, w, o in zip(xs, ws, outs): fn(x, w, o)
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1e3 / len(ws))
    return statistics.median(ts)

SHAPES = {'qkvr_local': (8600, 2304), 'qside_global': (6552, 2304), 'kv_global': (1024, 2304), 'wo': (2304, 3072),
          'lm_head': (201024, 2304), 'draft_gu': (18432, 2304), 'draft_down': (2304, 9216)}
CFGS = [(16,512,4,3),(16,512,4,2),(16,512,8,3),(16,512,2,3),(16,1024,4,2),(16,1024,8,2),(32,512,4,2),(32,512,8,2),(16,256,2,4),(16,512,4,4)]
only = sys.argv[1:] or list(SHAPES)
res = {}
for name in only:
    n, k = SHAPES[name]
    cnt = max(2, min(40, int(600e6 // (n * k * 2))))
    ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(cnt)]
    m = 8
    xs = [torch.randn(m, k, device=dev, dtype=torch.bfloat16) for _ in range(cnt)]
    od = torch.float32 if name == 'lm_head' else torch.bfloat16
    ref = [torch.empty(m, n, device=dev, dtype=od) for _ in range(cnt)]
    t = timeit(stock, xs, ws, ref); mb = n * k * 2 / 1e6
    print(f'{name:12s} N={n} K={k} x{cnt}  stock {t:8.1f} us {mb/t*1e3:6.1f} GB/s', flush=True)
    for cfg in CFGS:
        outs = [torch.empty_like(r) for r in ref]
        try:
            tv = timeit(run_v(cfg), xs, ws, outs)
        except Exception as e:
            print('   cfg', cfg, 'ERR', str(e)[:80]); continue
        exact = all(torch.equal(a, b) for a, b in zip(outs, ref))
        print(f'   cfg {str(cfg):18s} {tv:8.1f} us {mb/tv*1e3:6.1f} GB/s  exact={exact}', flush=True)
    del ws
