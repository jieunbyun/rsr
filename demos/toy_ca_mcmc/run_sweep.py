"""The sweep of rsr.mcmc (box jumps under component independence) on bn-on-demand's toy-ca.

Cases: each origin n3..n8 observed failed alone, then n4 + n6 and n4 + n6 + n7 observed failed
together. Every one of the 1024 edge configurations is enumerated, so each case is checked
against its exact posterior on the certified event E (the states every observed system's
failure boxes cover):
    P(E)          exact, and from the sequential start (Lemma 2); with one system also
                  sum_r P(L^r) / mean c (Eq. 15)
    closed err    max over edges of |P(e closed | E) - exact|
    TV            total variation distance over the 1024 states
    occupancy     max over boxes of |box frequency - pi(lambda_j = r)| (Eq. 10)
    mean c        chain average of c(x) against E_pi[c]
    rho1, ESS, R-hat   over the indicators 1(e closed)
The old results (results.json, results_multi.json: 2 chains, 10,000 kept steps or sweeps, the
previous kernels) are shown alongside. Writes results_sweep.json and run_sweep.log.

    PYTHONPATH=. python demos/toy_ca_mcmc/run_sweep.py
"""
import json
from pathlib import Path

import torch

from rsr import mcmc

N_CHAIN, N_BURN, N_KEEP = 1000, 200, 2000
CASES = [(o,) for o in ("n3", "n4", "n5", "n6", "n7", "n8")] + [("n4", "n6"), ("n4", "n6", "n7")]
HERE = Path(__file__).resolve().parent
data = torch.load(HERE / "toy_ca.pt")
probs, edges = data["probs"], data["edges"]
states = (torch.arange(1024)[:, None] >> torch.arange(9, -1, -1)) & 1   # (1024, 10), as exported
p_x = probs[torch.arange(10), states].prod(dim=1)
old = {**{k: v["closed_err"] for k, v in json.loads((HERE / "results.json").read_text()).items()},
       **{k: v["closed_err"] for k, v in json.loads((HERE / "results_multi.json").read_text()).items()}}
lines = []


def log(s):
    print(s, flush=True)
    lines.append(s)


def exact(mdl):
    chi = mcmc.membership(mdl, states)
    c = mcmc.counts(chi).double()
    inside = (c >= 1).all(dim=1)
    pi = p_x * inside / (p_x * inside).sum()
    return {"P_E": (p_x * inside).sum().item(), "pi": pi, "closed": (pi[:, None] * (states == 0)).sum(dim=0),
            "occupancy": [(pi[:, None] * ch / c[:, j:j + 1].clamp(min=1)).sum(dim=0) for j, ch in enumerate(chi)],
            "mean_c": (pi * c.prod(dim=1)).sum().item()}


log(f"{N_CHAIN} chains, burn-in {N_BURN} sweeps, {N_KEEP} kept sweeps, sequential start, on cpu\n")
log("case        R          P(E)     P_init/P  P_c/P   closed err (old)   TV      accept        "
    "label chg     occ err  mean c/E  rho1 max  ESS min   R-hat max  ms/sweep")
results = {}
for case in CASES:
    name = "+".join(case)
    mdl = mcmc.model([data["refs_lower"][o] for o in case], probs)
    ex = exact(mdl)
    tr = mcmc.run(mdl, N_CHAIN, N_BURN + N_KEEP, N_BURN, generator=torch.Generator().manual_seed(0))

    code = (tr["x"].long() * 2 ** torch.arange(9, -1, -1)).sum(dim=-1).flatten()
    freq = torch.bincount(code, minlength=1024).double() / len(code)
    mx = mcmc.mixing(tr, 2)
    var = mx["mean"][:, 0] * (1 - mx["mean"][:, 0]) > 0                   # edges not fixed by every box
    occ = mcmc.occupancy(tr, mdl)
    c_mean, cj_mean = mcmc.mean_counts(tr)
    P_c = None
    if len(case) == 1:                                                     # Z / mean c, Eq. (15)
        n, M = probs.shape
        Z = mcmc.label_logprob(mdl, 0, torch.zeros(1, n, dtype=torch.long),
                               torch.full((1, n), M - 1)).exp().sum().item()
        P_c = Z / c_mean
    ms = mcmc.timing(tr)
    r = {"systems": list(case), "n_refs": [len(b.lo) for b in mdl.systems], "P_E": ex["P_E"],
         "P_init": tr["p_event"], "P_counts": P_c,
         "closed_mcmc": mx["mean"][:, 0].tolist(), "closed_exact": ex["closed"].tolist(),
         "closed_err": (mx["mean"][:, 0] - ex["closed"]).abs().max().item(), "closed_err_old": old.get(name),
         "tv": 0.5 * (freq - ex["pi"]).abs().sum().item(),
         "acceptance": mcmc.acceptance_rate(tr).tolist(), "label_change": mcmc.label_change_rate(tr).tolist(),
         "occupancy_err": max((o - e).abs().max().item() for o, e in zip(occ, ex["occupancy"])),
         "mean_c": c_mean, "mean_c_exact": ex["mean_c"], "mean_cj": cj_mean.tolist(),
         "rho1_max": mx["rho1"][var, 0].max().item() if var.any() else 0.0,
         "ess_min": mx["ess"][var, 0].min().item() if var.any() else float(N_CHAIN * N_KEEP),
         "rhat_max": mx["rhat"][var, 0].max().item() if var.any() else 1.0,
         "ms": ms}
    results[name] = r
    pc = f"{P_c / ex['P_E']:.4f}" if P_c else "  -   "
    log(f"{name:10s}  {str(r['n_refs']):10s} {ex['P_E']:.5f}  {tr['p_event'] / ex['P_E']:.4f}    {pc}  "
        f"{r['closed_err']:.4f} ({r['closed_err_old']:.4f})    {r['tv']:.4f}  "
        f"{' '.join(f'{a:.3f}' for a in r['acceptance']):12s}  "
        f"{' '.join(f'{a:.3f}' for a in r['label_change']):12s}  {r['occupancy_err']:.4f}   "
        f"{c_mean / ex['mean_c']:.4f}    {r['rho1_max']:+.3f}    {r['ess_min']:8.0f}  {r['rhat_max']:.4f}     "
        f"{ms['sweep_median']:6.2f}")

log("\nms per sweep by step (1000 chains, all systems of a sweep):")
log("case        " + "  ".join(f"{k:>13s}" for k in mcmc.STEPS))
for name, r in results.items():
    log(f"{name:10s}  " + "  ".join(f"{r['ms'][k]:13.3f}" for k in mcmc.STEPS))

(HERE / "results_sweep.json").write_text(json.dumps(results, indent=2) + "\n")
(HERE / "run_sweep.log").write_text("\n".join(lines) + "\n")
print(f"\nwrote {HERE / 'results_sweep.json'} and run_sweep.log")
