"""Joint failure of every pair of EMA origins, at any probability level, by weighted IS.

For each origin a, draw x from its failure-union posterior (box r with probability P(L^r) /
sum P(L^r), x ~ p(. | L^r), weight 1 / c_a(x)) and check whether x lies in each origin b's
failure union:
    P(b | a) = E_q[1(x in U_b) / c_a] / E_q[1 / c_a],     P(a and b) = P(a) P(b | a)
Writes pairs_is.pt with P (K,), P_cond (K, K) = P(b | a) and P_both (K, K).

    PYTHONPATH=. python demos/ema_mcmc/pairs_is.py --n 200000
"""
import argparse
import time
from pathlib import Path

import torch

from rsr import mcmc

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=200_000)
parser.add_argument("--chunk", type=int, default=20_000)
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
data, ref = torch.load(HERE / "ema.pt"), torch.load(HERE / "reference_is.pt")
O, probs = data["origins"], data["probs"].to(dev)
low = {o: data["refs_lower"][o].to(dev) for o in O}
g = torch.Generator(device=dev).manual_seed(1)

K = len(O)
P = torch.tensor([ref["P_union"][o] for o in O], dtype=torch.float64)
P_cond = torch.zeros(K, K, dtype=torch.float64)
t0 = time.perf_counter()
for a, o in enumerate(O):
    p_box = (probs[None] * low[o]).sum(dim=2).prod(dim=1)
    num, den = torch.zeros(K, dtype=torch.float64, device=dev), torch.zeros((), dtype=torch.float64, device=dev)
    for _ in range(args.n // args.chunk):
        r = torch.multinomial(p_box, args.chunk, replacement=True, generator=g)
        x = mcmc._draw(probs[None] * low[o][r], g)
        w = 1.0 / mcmc._membership(x, low[o]).sum(dim=1).double()
        den += w.sum()
        for b, ob in enumerate(O):
            num[b] += (w * mcmc._membership(x, low[ob]).any(dim=1)).sum()
    P_cond[a] = (num / den).cpu()
    print(f"\r{a + 1}/{K} origins, {time.perf_counter() - t0:.0f} s", end="", flush=True)
print()
torch.save({"origins": O, "P": P, "P_cond": P_cond, "P_both": P[:, None] * P_cond}, HERE / "pairs_is.pt")
print("wrote", HERE / "pairs_is.pt")
