"""Independent reference for every EMA origin, at any failure probability: draw a box r with
probability P(L^r) / sum_r P(L^r), then x ~ p(. | x in L^r). The draws follow
q(x) = p(x) c(x) / sum_r P(L^r), so weights 1/c(x) give the posterior on the failure union:
    P(union)                 = sum_r P(L^r) * E_q[1 / c]
    P(e closed | union)      = E_q[1(e closed) / c] / E_q[1 / c]
Standard errors come from the spread between batches. Writes reference_is.pt.

    PYTHONPATH=. python demos/ema_mcmc/reference_is.py --n 1000000
"""
import argparse
import time
from pathlib import Path

import torch

from rsr import mcmc

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1_000_000, help="draws per origin")
parser.add_argument("--batches", type=int, default=20)
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.load(HERE / "ema.pt")
probs = data["probs"].to(dev)
g = torch.Generator(device=dev).manual_seed(0)

out = {"origins": data["origins"], "n_draws": args.n, "P_union": {}, "P_union_se": {}, "closed": {},
       "closed_se": {}, "sum_P_box": {}}
t0 = time.perf_counter()
for o in data["origins"]:
    refs = data["refs_lower"][o].to(dev)
    p_box = (probs[None] * refs).sum(dim=2).prod(dim=1)                    # (R,) P(L^r)
    w0, w1 = [], []                                                        # per batch: E[1/c], E[f/c]
    m = args.n // args.batches
    for _ in range(args.batches):
        r = torch.multinomial(p_box, m, replacement=True, generator=g)
        x = mcmc._draw(probs[None] * refs[r], g)                           # (m, n, 2)
        inv_c = 1.0 / mcmc._membership(x, refs).sum(dim=1).double()
        w0.append(inv_c.mean())
        w1.append((inv_c[:, None] * x[:, :, 0].double()).mean(dim=0))
    w0, w1 = torch.stack(w0), torch.stack(w1)                              # (B,), (B, n)
    P_b, closed_b = p_box.sum() * w0, w1 / w0[:, None]
    B = args.batches
    out["sum_P_box"][o] = p_box.sum().item()
    out["P_union"][o] = P_b.mean().item()
    out["P_union_se"][o] = (P_b.std() / B ** 0.5).item()
    out["closed"][o] = (w1.mean(dim=0) / w0.mean()).cpu()
    out["closed_se"][o] = (closed_b.std(dim=0) / B ** 0.5).cpu()
torch.save(out, HERE / "reference_is.pt")
print(f"{len(data['origins'])} origins x {args.n:,} draws in {time.perf_counter() - t0:.0f} s -> "
      f"{HERE / 'reference_is.pt'}")
