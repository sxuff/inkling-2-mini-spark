# Accuracy (rel L2, softcapped-softmax KL, top-1) and speed of the 8-bit GEMV on the real LM head and Markov fold.
# Run inside the image with the repo at /w and the checkpoint at /model.
import importlib.util, torch, statistics, json
from safetensors import safe_open
spec = importlib.util.spec_from_file_location('gq8', '/w/overlay/vllm/models/inkling2/nvidia/ops/gemv_q8.py'); gq8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq8)
dev = 'cuda'; torch.manual_seed(0)
with safe_open('/model/model-00001-of-00001.safetensors', 'pt', device='cuda') as f:
    lm = f.get_tensor('model.llm.unembed.weight'); emb = f.get_tensor('model.llm.embed.weight')
with safe_open('/model/mtp.safetensors', 'pt', device='cuda') as f:
    keys = list(f.keys()); ho = [k for k in keys if 'hidden_out' in k]; print('hidden_out keys', ho)
    how = f.get_tensor([k for k in ho if k.endswith('weight')][0]).double()
print('lm', lm.shape, lm.dtype)
fold = torch.cat([(lm[i:i+8192].double() @ how).to(torch.bfloat16) for i in range(0, lm.shape[0], 8192)])
V = 200058
def metrics(ref, got):
    ref = ref[:, :V].float(); got = got[:, :V].float()
    rel = ((got - ref).norm() / ref.norm()).item()
    cap = lambda z: 30 * torch.tanh(z / 30)
    p = torch.log_softmax(cap(ref), -1); q = torch.log_softmax(cap(got), -1)
    kl = (p.exp() * (p - q)).sum(-1).mean().item()
    top1 = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
    return rel, kl, top1
# realistic-ish hidden: final norm'd hidden ~ unit RMS; use rows correlated with embeddings + noise
idx = torch.randint(0, V, (64,), device=dev)
xs = {'emb_like': (emb[idx].float() / emb[idx].float().pow(2).mean(-1, keepdim=True).sqrt() * 1.0).to(torch.bfloat16),
      'gauss': torch.randn(64, 2304, device=dev, dtype=torch.bfloat16)}
for mode in ('int8', 'fp8'):
    q = gq8.quantize_rows(lm, mode)
    for name, x in xs.items():
        for m in (7, 56):
            ref = (x[:m].float() @ lm.float().T)
            got = gq8.gemv_q8(x[:m], *q)
            print(mode, 'lm_head', name, 'M=%d' % m, 'rel %.4f KL %.5f top1 %.3f' % metrics(ref, got))
    fq = gq8.quantize_rows(fold, mode)
    lat = torch.tanh(torch.randn(8, 256, device=dev)).to(torch.bfloat16)
    ref = lat.float() @ fold.float().T; got = gq8.gemv_q8(lat[:1], *fq)
    print(mode, 'fold rel %.4f' % ((got.float() - ref[:1]).norm() / ref[:1].norm()).item())
    del q, fq
# timing of dispatcher
def timeit(fn, reps=20):
    s = torch.cuda.Stream(); g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)
q = gq8.quantize_rows(lm, 'int8'); fq = gq8.quantize_rows(fold, 'int8'); bias = torch.zeros(lm.shape[0], device=dev, dtype=torch.bfloat16)
for m in (7, 14, 56):
    x = xs['gauss'][:m]; print('lm_head int8 M=%d %.1f us' % (m, timeit(lambda: gq8.gemv_q8(x, *q))))
print('fold int8 M=1 %.1f us' % timeit(lambda: gq8.gemv_q8(lat[:1], *fq, bias=bias)))
print('fold bf16 addmm M=1 %.1f us' % timeit(lambda: torch.addmm(bias, lat[:1], fold.T)))
