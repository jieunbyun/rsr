"""rsr.mcmc on bn-on-demand's EMA highways (129 binary edges), with two systems observed: n57
and n58 both failed. Their failure boxes (2451 and 2665) overlap in a few small regions, so a
chain that only moves one system's box at a time rarely crosses between them; the tuple jump
moves both boxes at once.

Data: ema_n57_n58.pt (export_ema.py --origins n57 n58): the edges' closure probabilities and
the two origins' reference states. The model is built with mcmc.observe; the event E is the set
of edge states that both failure-box sets cover.

Reference: weighted importance sampling from the intersection of the two systems' boxes. A box
pair (r1, r2) is drawn with probability P(L^{1,r1} n L^{2,r2}) / Z12, then x ~ p(. | L^{1,r1} n
L^{2,r2}); the density is p(x) c_1(x) c_2(x) / Z12, so every draw is in E with weight
1 / (c_1(x) c_2(x)), and P(E) = Z12 * mean weight. N_REF draws, cached in reference.pt; the
standard errors come from N_GROUP batches.

MCMC: N_CHAIN chains from the sequential start, N_BURN + N_KEEP sweeps, each followed by a tuple
jump (N_PARTNER partner tuples from rho of Eq. (rho) with gamma = GAMMA, systems in a random order
per jump), recorded from the first sweep.

Writes run.log, results.json (summary, read by plot.py) and traces.pt (the boxes of every chain,
the share of chains with each edge closed and the edge states of the first N_TRACE chains, per
sweep; read by plot.py, not committed).

    PYTHONPATH=. python demos/ema_mcmc/run.py
    python demos/ema_mcmc/plot.py
"""
import json
import math
import time
from pathlib import Path

import torch

from rsr import mcmc

OBSERVED = {"n57": 0, "n58": 0}          # 0 = failed, 1 = survived
N_CHAIN, N_BURN, N_KEEP, N_PARTNER, GAMMA, N_TRACE = 1000, 300, 1000, 8, 1.0, 2
N_REF, BATCH, N_GROUP, ROW_CHUNK = 4_000_000, 100_000, 20, 32
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.load(HERE / "ema_n57_n58.pt")
probs, edges = data["probs"].to(dev), data["edges"]
rules = {o: (data["refs_upper"][o], data["refs_lower"][o]) for o in data["origins"]}
mdl = mcmc.observe(rules, OBSERVED, probs)
n, M = probs.shape
lines = []


def log(s):
    print(s, flush=True)
    lines.append(s)


def pair_logprob():
    """log P(L^{1,r1} n L^{2,r2}) for every box pair, (R1, R2)."""
    b1, b2 = mdl.systems
    comp = torch.arange(n, device=dev)
    out = torch.empty(len(b1.lo), len(b2.lo), dtype=torch.float64, device=dev)
    for s in range(0, len(b1.lo), ROW_CHUNK):
        lo = torch.maximum(b1.lo[s:s + ROW_CHUNK, None], b2.lo[None])
        hi = torch.minimum(b1.hi[s:s + ROW_CHUNK, None], b2.hi[None])
        lp = mcmc._mass(mdl.cdf, comp.expand_as(lo), lo, torch.maximum(lo, hi)).log().sum(dim=-1)
        out[s:s + ROW_CHUNK] = torch.where((lo > hi).any(dim=-1), -torch.inf, lp)
    return out


def reference():
    path = HERE / "reference.pt"
    if path.exists() and (ref := torch.load(path))["n_draw"] == N_REF:
        return ref
    t0 = time.perf_counter()
    g = torch.Generator(device=dev).manual_seed(2)
    lp = pair_logprob().flatten()
    b1, b2 = mdl.systems
    R2 = len(b2.lo)
    w_cat = (lp - lp.max()).exp()
    num, den, w2 = [], [], 0.0
    for _ in range(N_REF // BATCH):
        idx = torch.multinomial(w_cat, BATCH, replacement=True, generator=g)
        r1, r2 = idx // R2, idx % R2
        x = mcmc.draw_state(mdl, torch.maximum(b1.lo[r1], b2.lo[r2]), torch.minimum(b1.hi[r1], b2.hi[r2]), g)
        c = mcmc.counts(mcmc.membership(mdl, x)).double()
        w = 1 / c.prod(dim=1)
        num.append((w[:, None] * (x == 0)).sum(dim=0))
        den.append(w.sum())
        w2 += (w ** 2).sum().item()
    num, den = torch.stack(num), torch.stack(den)
    k = len(den) // N_GROUP
    closed_b = num[:k * N_GROUP].view(N_GROUP, k, n).sum(1) / den[:k * N_GROUP].view(N_GROUP, k).sum(1)[:, None]
    D = den.sum().item()
    ref = {"closed": (num.sum(0) / D).cpu(), "closed_se": (closed_b.std(0) / N_GROUP ** 0.5).cpu(),
           "P": math.exp(torch.logsumexp(lp, 0).item()) * D / N_REF, "ess": D ** 2 / w2, "n_draw": N_REF,
           "sec": time.perf_counter() - t0}
    torch.save(ref, path)
    return ref


ref = reference()
obs = ", ".join(f"{o} {'failed' if OBSERVED[o] == 0 else 'survived'}" for o in mdl.names)
log(f"EMA, {obs}; boxes {[len(b.lo) for b in mdl.systems]}; on {dev}")
log(f"reference: {ref['n_draw']:,} weighted-IS draws from the box-pair intersections, P(E) = {ref['P']:.3e}, "
    f"ESS {ref['ess']:,.0f}, {ref['sec']:.0f} s")

g = torch.Generator(device=dev).manual_seed(0)
t0 = time.perf_counter()
full = mcmc.run(mdl, N_CHAIN, N_BURN + N_KEEP, 0, generator=g, n_partner=N_PARTNER, rho=mcmc.partner_rho(mdl, GAMMA))
wall = time.perf_counter() - t0
per_sweep = ("x", "t", "c", "ok", "changed", "tuple_ok", "tuple_changed")
tr = {**full, **{k: full[k][N_BURN:] for k in per_sweep}, "n_burn": N_BURN}

closed = (tr["x"] == 0).double().mean(dim=(0, 1)).cpu()
err = (closed - ref["closed"].double()).abs()
mx = mcmc.mixing(tr, 2)
var = mx["mean"][:, 0] * (1 - mx["mean"][:, 0]) > 0
acc, moved = mcmc.tuple_rates(tr)
ms = mcmc.timing(tr)
r = {"title": "EMA", "systems": mdl.names, "observed": [OBSERVED[o] for o in mdl.names],
     "n_refs": [len(b.lo) for b in mdl.systems], "n_chain": N_CHAIN, "n_burn": N_BURN, "n_keep": N_KEEP,
     "n_partner": N_PARTNER, "gamma": GAMMA, "ref_name": "weighted-IS", "P_E": ref["P"], "ess_ref": ref["ess"],
     "P_init": full["p_event"], "closed_err": err.max().item(), "n_off": int((err > 0.05).sum()),
     "rhat_max": mx["rhat"][var, 0].max().item(), "ess_min": mx["ess"][var, 0].min().item(),
     "acceptance": mcmc.acceptance_rate(tr).tolist(), "label_change": mcmc.label_change_rate(tr).tolist(),
     "tuple_accept": acc, "tuple_moved": moved, "occupied": [int((o > 0).sum()) for o in mcmc.occupancy(tr, mdl)],
     "wall_sec": wall, "ms": ms, "edges": edges, "closed_mcmc": closed.tolist(),
     "closed_ref": ref["closed"].tolist(), "closed_ref_se": ref["closed_se"].tolist()}

log(f"{N_CHAIN} chains, {N_BURN} + {N_KEEP} sweeps, each followed by a tuple jump (m = {N_PARTNER}, "
    f"gamma = {GAMMA:g}, random order), sequential start; {wall:.0f} s")
log(f"P(e closed | E): max error {r['closed_err']:.4f} against the reference ({r['n_off']}/{n} edges off by > 0.05)")
log(f"R-hat max {r['rhat_max']:.3f}, ESS min {r['ess_min']:.0f}; P(E) from the sequential start / reference "
    f"{r['P_init'] / r['P_E']:.3f}")
log(f"sweep: accept {[round(a, 3) for a in r['acceptance']]}, box change {[round(a, 3) for a in r['label_change']]}; "
    f"tuple jump: accept {acc:.3f}, to a partner {moved:.4f} of accepted; boxes occupied {r['occupied']} of "
    f"{r['n_refs']}")
log("ms per sweep (all chains): " + ", ".join(f"{k} {ms[k]:.2f}" for k in mcmc.STEPS + mcmc.TUPLE_STEPS)
    + f"; median sweep {ms['sweep_median']:.1f}")

(HERE / "results.json").write_text(json.dumps(r, indent=2) + "\n")
torch.save({"t": full["t"].short().cpu(), "x": full["x"][:, :N_TRACE].cpu(),
            "tuple_changed": full["tuple_changed"][:, :N_TRACE].cpu(),
            "closed_share": (full["x"] == 0).float().mean(dim=1).cpu(), "n_burn": N_BURN}, HERE / "traces.pt")
(HERE / "run.log").write_text("\n".join(lines) + "\n")
print(f"wrote run.log, results.json and traces.pt in {HERE}")
