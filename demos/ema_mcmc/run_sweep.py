"""The sweep of rsr.mcmc (box jumps under component independence) on bn-on-demand's EMA highways
(129 binary edges), with the cases of run_bridge.py and run_pair.py:
    single  the five origins with the most failure reference states, each observed failed,
            against the weighted-IS reference (reference_is.pt)
    pair    n57 and n58 observed failed together, against weighted IS on n58's failure boxes
            keeping the draws that also fail n57 (cached in reference_pair.pt)
Each case starts from the sequential draw of Section 11 and from the old worst start (chain k at
the upper corner of box k mod R_j of every system, so every box starts with equal weight).
Standard errors come from 20 groups of chains:
    closed err   max over edges of |P(e closed | E) - IS|,  z2 = mean of ((MCMC - IS) / SE)^2
    P_init/P     P(E) from the sequential start (Lemma 2), P_c/P from sum_r P(L^r) / mean c
    occupied     reference boxes occupied at least once, of R_j per system
The old errors (results_bridge.json, results_pair.json) are shown alongside. Writes
results_sweep.json and run_sweep.log.

    PYTHONPATH=. python demos/ema_mcmc/run_sweep.py
"""
import json
import time
from pathlib import Path

import torch

from rsr import mcmc

N_CHAIN, N_BURN, N_KEEP, N_GROUP, N_TOP = 1000, 300, 1000, 20, 5
N_REF, REF_BATCH = 4_000_000, 50_000
PAIR = ("n57", "n58")
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
data, ref_single = torch.load(HERE / "ema.pt"), torch.load(HERE / "reference_is.pt")
probs, edges = data["probs"].to(dev), data["edges"]
n, M = probs.shape
lines = []


def log(s):
    print(s, flush=True)
    lines.append(s)


def box_logprob(mdl, j):
    """log P(L^{j,r}) for every box of system j."""
    return mcmc.label_logprob(mdl, j, torch.zeros(1, n, dtype=torch.long, device=dev),
                              torch.full((1, n), M - 1, device=dev))[0]


def reference_pair():
    """P(e closed | n57 and n58 fail): box r of n58 with probability P(L^r), x ~ p(. | L^r),
    weight 1(x fails n57) / c_58(x)."""
    path = HERE / "reference_pair.pt"
    if path.exists():
        return torch.load(path)
    t0 = time.perf_counter()
    m58, m57 = mcmc.model([data["refs_lower"]["n58"]], probs), mcmc.model([data["refs_lower"]["n57"]], probs)
    lp = box_logprob(m58, 0)
    g = torch.Generator(device=dev).manual_seed(1)
    num, den = [], []
    for _ in range(N_REF // REF_BATCH):
        r = torch.multinomial((lp - lp.max()).exp(), REF_BATCH, replacement=True, generator=g)
        bx = m58.systems[0]
        x = mcmc.draw_state(m58, bx.lo[r], bx.hi[r], g)
        w = mcmc.membership(m57, x)[0].any(dim=1) / mcmc.counts(mcmc.membership(m58, x))[:, 0].double()
        num.append((w[:, None] * (x == 0)).sum(dim=0))
        den.append(w.sum())
    num, den = torch.stack(num), torch.stack(den)
    B = len(den)
    k = B // N_GROUP
    closed_b = num.view(N_GROUP, k, n).sum(dim=1) / den.view(N_GROUP, k).sum(dim=1)[:, None]
    out = {"closed": (num.sum(0) / den.sum()).cpu(), "closed_se": (closed_b.std(0) / N_GROUP ** 0.5).cpu(),
           "P": (lp.exp().sum() * den.sum() / N_REF).item(), "n_draw": N_REF,
           "sec": time.perf_counter() - t0}
    torch.save(out, path)
    return out


def worst_start(mdl, C):
    """Chain k at the upper corner of the intersection of box k mod R_j of every system."""
    t = torch.stack([torch.arange(C, device=dev) % len(b.lo) for b in mdl.systems], dim=1)
    hi = torch.full((C, n), M - 1, device=dev)
    for j, b in enumerate(mdl.systems):
        hi = torch.minimum(hi, b.hi[t[:, j]])
    return hi, t


def evaluate(name, systems, ref_closed, ref_se, P_ref, init, old_err):
    mdl = mcmc.model([data["refs_lower"][o] for o in systems], probs)
    g = torch.Generator(device=dev).manual_seed(0)
    start = worst_start(mdl, N_CHAIN) if init == "worst" else None
    tr = mcmc.run(mdl, N_CHAIN, N_BURN + N_KEEP, N_BURN, init=start, generator=g)
    closed_chain = (tr["x"] == 0).double().mean(dim=0)                      # (C, n)
    groups = torch.arange(N_CHAIN, device=dev) % N_GROUP
    closed = closed_chain.mean(dim=0)
    closed_se = torch.stack([closed_chain[groups == k].mean(dim=0) for k in range(N_GROUP)]).std(dim=0) / N_GROUP ** 0.5
    ref_closed, ref_se = ref_closed.to(dev).double(), ref_se.to(dev).double()
    se = (closed_se ** 2 + ref_se ** 2).sqrt()
    z = (closed - ref_closed)[se > 0] / se[se > 0]
    mx = mcmc.mixing(tr, 2)
    var = mx["mean"][:, 0] * (1 - mx["mean"][:, 0]) > 0
    occ = mcmc.occupancy(tr, mdl)
    c_mean, cj_mean = mcmc.mean_counts(tr)
    P_c = None
    if len(systems) == 1:
        P_c = box_logprob(mdl, 0).exp().sum().item() / c_mean
    ms = mcmc.timing(tr)
    r = {"systems": list(systems), "init": init, "n_refs": [len(b.lo) for b in mdl.systems], "n_chain": N_CHAIN,
         "P_ref": P_ref, "P_init": tr["p_event"], "P_counts": P_c,
         "closed_err": (closed - ref_closed).abs().max().item(), "closed_err_old": old_err,
         "z2": (z ** 2).mean().item() if len(z) else 0.0,
         "acceptance": mcmc.acceptance_rate(tr).tolist(), "label_change": mcmc.label_change_rate(tr).tolist(),
         "occupied": [int((o > 0).sum()) for o in occ], "mean_c": c_mean, "mean_cj": cj_mean.tolist(),
         "rho1_max": mx["rho1"][var, 0].max().item(), "ess_min": mx["ess"][var, 0].min().item(),
         "rhat_max": mx["rhat"][var, 0].max().item(), "ms": ms,
         "closed_mcmc": closed.tolist(), "closed_ref": ref_closed.tolist()}
    p_init = f"{tr['p_event'] / P_ref:.4f}" if tr["p_event"] else "  -   "
    p_c = f"{P_c / P_ref:.4f}" if P_c else "  -   "
    old = f"{old_err:.4f}" if old_err is not None else "  -   "
    log(f"{name:8s} {init:10s} {str(r['n_refs']):12s} {P_ref:.3e}  {p_init}  {p_c}  {r['closed_err']:.4f} "
        f"({old})  {r['z2']:7.2f}  {' '.join(f'{a:.3f}' for a in r['acceptance']):12s} "
        f"{' '.join(f'{a:.3f}' for a in r['label_change']):12s} "
        f"{'/'.join(str(o) for o in r['occupied']):10s} {c_mean:6.2f}  {r['rho1_max']:.3f}  "
        f"{r['ess_min']:8.0f}  {r['rhat_max']:7.3f}  {ms['sweep_median']:7.1f}")
    return r


old_single = json.loads((HERE / "results_bridge.json").read_text())
old_pair = json.loads((HERE / "results_pair.json").read_text())
pair_ref = reference_pair()
log(f"{N_CHAIN} chains, burn-in {N_BURN} sweeps, {N_KEEP} kept sweeps, on {dev}; pair reference: "
    f"{pair_ref['n_draw']:,} weighted-IS draws, P(both fail) = {pair_ref['P']:.3e}")
log("old error: Kernel I and Kernel I + bridge m=8, N=128 (one chain per box, run_bridge.py); "
    "pair: step_multi and + bridge beta=0.1 (run_pair.py)\n")
log("case     start      R            P_ref      P_init/P P_c/P   closed err (old)  z2       accept       "
    "label chg    occupied   mean c rho1   ESS min   R-hat    ms/sweep")
results = {}
origins = sorted(data["origins"], key=lambda o: -len(data["refs_lower"][o]))[:N_TOP]
for o in origins:
    for init in ("sequential", "worst"):
        old_err = old_single["Kernel I"][o]["closed_err"] if init == "worst" else None
        results[f"{o}/{init}"] = evaluate(o, (o,), ref_single["closed"][o], ref_single["closed_se"][o],
                                          ref_single["P_union"][o], init, old_err)
for init in ("sequential", "worst"):
    old_err = old_pair["step_multi"]["closed_err"] if init == "worst" else None
    results[f"{'+'.join(PAIR)}/{init}"] = evaluate("+".join(PAIR), PAIR, pair_ref["closed"], pair_ref["closed_se"],
                                                   pair_ref["P"], init, old_err)
log("\nold errors with the bridge: " + ", ".join(
    f"{o} {old_single['I + bridge m=8, N=128'][o]['closed_err']:.4f}" for o in origins)
    + f"; pair {old_pair['+ bridge m=8, N=128, overlap partners (beta=0.1)']['closed_err']:.4f}")

log("\nms per sweep by step (1000 chains, all systems of a sweep), and the start in ms:")
log("case                 " + "  ".join(f"{k:>13s}" for k in mcmc.STEPS) + "           init")
for name, r in results.items():
    log(f"{name:20s} " + "  ".join(f"{r['ms'][k]:13.3f}" for k in mcmc.STEPS) + f"  {r['ms']['init']:13.1f}")

(HERE / "results_sweep.json").write_text(json.dumps(results, indent=2) + "\n")
(HERE / "run_sweep.log").write_text("\n".join(lines) + "\n")
print(f"\nwrote {HERE / 'results_sweep.json'} and run_sweep.log")
