"""Direct Monte Carlo reference for the EMA MCMC runs (2^129 states cannot be enumerated).

Draws edge states from the exact independent prior (ema.pt), classifies every draw against
each origin's survival and failure boxes with rsr, and keeps for each origin:
    P_L, P_U, P_unknown : probability of the failure union, the survival union, neither
    hits                : draws in the failure union (the MCMC's target)
    closed              : (129,) P(e closed | failure union) by rejection, with its SE
and for every pair of origins P(both in their failure unions), from the same draws, to pick
correlated observations. Writes reference_mc.pt.

    PYTHONPATH=. python demos/ema_mcmc/reference_mc.py --n 4000000
"""
import argparse
import time
from pathlib import Path

import torch

from rsr.rsr import classify_samples_with_indices

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=4_000_000)
parser.add_argument("--chunk", type=int, default=200_000)
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.load(HERE / "ema.pt")
origins, p_closed = data["origins"], data["probs"][:, 0].to(dev)
up = {o: data["refs_upper"][o].to(dev) for o in origins}
low = {o: data["refs_lower"][o].to(dev) for o in origins}

K, n = len(origins), len(p_closed)
count = {k: torch.zeros(K, dtype=torch.float64, device=dev) for k in ("L", "U")}
closed_sum = torch.zeros(K, n, dtype=torch.float64, device=dev)
pair = torch.zeros(K, K, dtype=torch.float64, device=dev)
g = torch.Generator(device=dev).manual_seed(0)
t0 = time.perf_counter()
for start in range(0, args.n, args.chunk):
    m = min(args.chunk, args.n - start)
    x = (torch.rand(m, n, device=dev, generator=g) >= p_closed).long()      # 1 = open
    onehot = torch.nn.functional.one_hot(x, 2).int()
    L = torch.zeros(m, K, device=dev)
    for k, o in enumerate(origins):
        res = classify_samples_with_indices(onehot, up[o], low[o], return_masks=True)
        L[:, k] = res["mask_lower"].float()
        count["U"][k] += res["mask_upper"].sum()
    count["L"] += L.sum(dim=0).double()
    closed_sum += (L.T @ (x == 0).float()).double()
    pair += (L.T @ L).double()
    print(f"\r{start + m:,} / {args.n:,} draws, {time.perf_counter() - t0:.0f} s", end="", flush=True)
print()

hits = count["L"]
closed = closed_sum / hits.clamp(min=1)[:, None]
out = {
    "origins": origins, "n_draws": args.n, "hits": hits.cpu(),
    "P_L": (hits / args.n).cpu(), "P_U": (count["U"] / args.n).cpu(),
    "closed": closed.cpu(), "closed_se": (closed * (1 - closed) / hits.clamp(min=1)[:, None]).sqrt().cpu(),
    "P_pair": (pair / args.n).cpu(),
}
out["P_unknown"] = 1 - out["P_L"] - out["P_U"]
torch.save(out, HERE / "reference_mc.pt")
print(f"wrote {HERE / 'reference_mc.pt'} in {time.perf_counter() - t0:.0f} s")
