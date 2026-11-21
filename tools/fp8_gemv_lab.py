# FP8 weight-only GEMV tile sweep on the draft shapes (random weights; run inside the image).
import torch, statistics, sys
import triton, triton.language as tl
dev = 'cuda'; torch.manual_seed(0)

@triton.jit
def gemv_fp8(X, W, S, B, Y, M, N, K, SX, SW, SY, HAS_BIAS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(W + cols[None, :].to(tl.int64) * SW + ks[:, None], (cols[None, :] < N) & (ks[:, None] < K), other=0.0)
        x = tl.load(X + rows[:, None] * SX + ks[None, :], (rows[:, None] < M) & (ks[None, :] < K), other=0.0)
        acc += tl.dot(x, w.to(tl.bfloat16))
    s = tl.load(S + cols, cols < N, other=0.0)
    acc = acc * s[None, :]
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0.0).to(tl.float32)[None, :]
    tl.store(Y + rows[:, None] * SY + cols[None, :], acc.to(Y.dtype.element_ty), (rows[:, None] < M) & (cols[None, :] < N))

def quant(w):
    s = w.float().abs().amax(1).clamp_min(1e-12) / 448.0
    q = (w.float() / s[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, s

def run(cfg, x, q, s, out, bias=None):
    BN, BK, nw, ns = cfg
    m, k = x.shape; n = q.shape[0]
    gemv_fp8[(triton.cdiv(n, BN),)](x, q, s, bias, out, m, n, k, x.stride(0), q.stride(0), out.stride(0),
        HAS_BIAS=bias is not None, BN=BN, BK=BK, MP=16, num_warps=nw, num_stages=ns)

def timeit(fn, reps=15):
    s = torch.cuda.Stream(); g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)

SHAPES = {'lm_head': (201024, 2304, 7), 'fold': (201024, 256, 1), 'draft_qkv': (5120, 2304, 8), 'draft_o': (2304, 3072, 8),
          'draft_gu': (18432, 2304, 8), 'draft_down': (2304, 9216, 8), 'kv_ctx': (2048, 2304, 8)}
CFGS = [(16, 512, 4, 3), (32, 512, 4, 3), (64, 256, 4, 3), (32, 1024, 4, 3), (16, 1024, 4, 3), (64, 512, 8, 3), (128, 256, 8, 3), (64, 256, 4, 4), (32, 256, 4, 4)]
for name, (n, k, m) in SHAPES.items():
    cnt = max(2, min(40, int(400e6 // (n * k))))
    ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(cnt)]
    qs = [quant(w) for w in ws]
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    ref = (x.float() @ (qs[0][0].float() * qs[0][1][:, None]).T)
    exact_bf16 = (x.float() @ ws[0].float().T)
    outs = [torch.empty(m, n, device=dev, dtype=torch.bfloat16) for _ in range(cnt)]
    print(f'{name:10s} N={n} K={k} M={m}  bf16 bytes/call {n*k*2/1e6:.1f}MB  quant rel err vs bf16: {((ref-exact_bf16).norm()/exact_bf16.norm()).item():.4f}', flush=True)
    for cfg in CFGS:
        try:
            run(cfg, x, qs[0][0], qs[0][1], outs[0])
            err = ((outs[0].float() - ref).abs().max() / ref.abs().max()).item()
            def f():
                for (q, s), o in zip(qs, outs): run(cfg, x, q, s, o)
            t = timeit(f) / cnt
            print(f'   {str(cfg):18s} {t:8.1f} us  {n*k/t/1e3:6.1f} GB/s  maxerr {err:.2e}', flush=True)
        except Exception as e:
            print('   ', cfg, 'ERR', str(e)[:100])
    del ws, qs
