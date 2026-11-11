"""The sweep of rsr.mcmc followed by a tuple jump on bn-on-demand's EMA highways, for n57 and
n58 observed failed together (the pair the sweep alone could not mix, run_sweep.py), against
the weighted-IS reference of run_sweep.py (reference_pair.pt, which must exist).

Every sweep ends with one tuple jump: N_PARTNER partners from rho of Eq. (rho) with gamma = 1
and a random order of the systems per jump. Starts: the sequential draw (Section 11) and the
worst start of run_sweep.py (chain k at the upper corner of box k mod R_j of every system).
The chains are recorded from the first sweep, and the statistics use the sweeps after N_BURN.
Writes results_tuple.json (read by plot_posterior.py), traces_tuple.pt (read by
plot_traces_tuple.py: the labels of every chain, the share of chains with each edge closed
per sweep, and the edge states of the first N_TRACE chains, burn-in included) and
run_tuple.log.

    PYTHONPATH=. python demos/ema_mcmc/run_tuple.py
"""
import json
import time
from pathlib import Path

import torch

from rsr import mcmc

N_CHAIN, N_BURN, N_KEEP, N_PARTNER, GAMMA, N_TRACE = 1000, 300, 1000, 8, 1.0, 2
PAIR = ("n57", "n58")
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.load(HERE / "ema.pt")
probs = data["probs"].to(dev)
n, M = probs.shape
pref = torch.load(HERE / "reference_pair.pt")
mdl = mcmc.model([data["refs_lower"][o] for o in PAIR], probs)
ref = pref["closed"].to(dev).double()
lines = []


def log(s):
    print(s, flush=True)
    lines.append(s)


def worst_start(C):
    t = torch.stack([torch.arange(C, device=dev) % len(b.lo) for b in mdl.systems], dim=1)
    hi = torch.full((C, n), M - 1, device=dev)
    for j, b in enumerate(mdl.systems):
        hi = torch.minimum(hi, b.hi[t[:, j]])
    return hi, t


log(f"{'+'.join(PAIR)}: {N_CHAIN} chains, {N_BURN} + {N_KEEP} sweeps on {dev}, each followed by a tuple jump "
    f"with m = {N_PARTNER}, gamma = {GAMMA}, random order")
log("(sweep alone, run_sweep.log: closed err 0.4322 sequential / 0.3548 worst, R-hat 2.1 / 2.9, 11 ms per sweep)")
results, traces = {}, {}
for start in ("sequential", "worst"):
    g = torch.Generator(device=dev).manual_seed(0)
    t0 = time.perf_counter()
    full = mcmc.run(mdl, N_CHAIN, N_BURN + N_KEEP, 0, init=worst_start(N_CHAIN) if start == "worst" else None,
                    generator=g, n_partner=N_PARTNER, rho=mcmc.partner_rho(mdl, GAMMA))
    wall = time.perf_counter() - t0
    per_sweep = ("x", "t", "c", "ok", "changed", "tuple_ok", "tuple_changed")
    tr = {**full, **{k: full[k][N_BURN:] for k in per_sweep}, "n_burn": N_BURN}
    closed = (tr["x"] == 0).double().mean(dim=(0, 1))
    err = (closed - ref).abs()
    mx = mcmc.mixing(tr, 2)
    var = mx["mean"][:, 0] * (1 - mx["mean"][:, 0]) > 0
    acc, moved = mcmc.tuple_rates(tr)
    occ = mcmc.occupancy(tr, mdl)
    ms = mcmc.timing(tr)
    r = {"systems": list(PAIR), "init": start, "n_refs": [len(b.lo) for b in mdl.systems], "n_chain": N_CHAIN,
         "n_partner": N_PARTNER, "gamma": GAMMA, "P_ref": pref["P"], "closed_err": err.max().item(),
         "rhat_max": mx["rhat"][var, 0].max().item(), "ess_min": mx["ess"][var, 0].min().item(),
         "rho1_max": mx["rho1"][var, 0].max().item(),
         "acceptance": mcmc.acceptance_rate(tr).tolist(), "label_change": mcmc.label_change_rate(tr).tolist(),
         "tuple_accept": acc, "tuple_moved": moved, "occupied": [int((o > 0).sum()) for o in occ],
         "wall_sec": wall, "ms": ms, "closed_mcmc": closed.tolist(), "closed_ref": ref.tolist()}
    results[f"{'+'.join(PAIR)}/{start}"] = r
    traces[start] = {"t": full["t"].short().cpu(), "tuple_changed": full["tuple_changed"].cpu(),
                     "closed_share": (full["x"] == 0).float().mean(dim=1).cpu(),
                     "x": full["x"][:, :N_TRACE].cpu(), "n_burn": N_BURN, "result": r}
    log(f"\n{start} start ({wall:.0f} s): closed err {r['closed_err']:.4f} ({int((err > 0.05).sum())}/{n} edges "
        f"> 0.05), R-hat max {r['rhat_max']:.3f}, ESS min {r['ess_min']:.0f}")
    log(f"   sweep: accept {[round(a, 3) for a in r['acceptance']]}, label change "
        f"{[round(a, 3) for a in r['label_change']]}; tuple jump: accept {acc:.3f}, from a partner {moved:.3f}; "
        f"boxes occupied {r['occupied']} of {r['n_refs']}")
    log("   ms per sweep: " + ", ".join(f"{k} {ms[k]:.2f}" for k in mcmc.STEPS + mcmc.TUPLE_STEPS)
        + f"; median sweep {ms['sweep_median']:.1f}")

(HERE / "results_tuple.json").write_text(json.dumps(results, indent=2) + "\n")
torch.save({"traces": traces, "edges": data["edges"], "systems": list(PAIR), "closed_ref": ref.cpu()},
           HERE / "traces_tuple.pt")
(HERE / "run_tuple.log").write_text("\n".join(lines) + "\n")
print(f"\nwrote results_tuple.json, traces_tuple.pt and run_tuple.log in {HERE}")
