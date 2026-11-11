"""rsr.mcmc on bn-on-demand's toy-ca (10 binary edges), with two systems observed: n4 and n6
both failed.

The model is built with mcmc.observe from every origin's reference states (upper = survival
boxes, lower = failure boxes) and the observed states; the event E is the set of edge states
that every observed system's boxes certify. All 1024 edge states are enumerated, so the chains
are checked against the exact posterior.

Each sweep updates every observed system's box and the state, and is followed by a tuple jump
(N_PARTNER partner tuples drawn from rho of Eq. (rho) with gamma = GAMMA, systems in a random
order per jump). N_CHAIN chains from the sequential start, N_BURN + N_KEEP sweeps, recorded from
the first sweep.

Writes run.log, results.json (summary, read by plot.py) and traces.pt (the boxes of every chain,
the share of chains with each edge closed and the edge states of the first N_TRACE chains, per
sweep; read by plot.py, not committed).

    PYTHONPATH=. python demos/toy_ca_mcmc/run.py
    python demos/toy_ca_mcmc/plot.py
"""
import json
import time
from pathlib import Path

import torch

from rsr import mcmc

OBSERVED = {"n4": 0, "n6": 0}            # 0 = failed, 1 = survived
N_CHAIN, N_BURN, N_KEEP, N_PARTNER, GAMMA, N_TRACE = 1000, 200, 2000, 8, 1.0, 2
HERE = Path(__file__).resolve().parent
data = torch.load(HERE / "toy_ca.pt")
probs, edges = data["probs"], data["edges"]
rules = {o: (data["refs_upper"][o], data["refs_lower"][o]) for o in data["origins"]}
mdl = mcmc.observe(rules, OBSERVED, probs)
n = len(edges)
lines = []


def log(s):
    print(s, flush=True)
    lines.append(s)


# exact posterior on E by enumeration; the exported state order is edge 0 as the highest bit
states = (torch.arange(2 ** n)[:, None] >> torch.arange(n - 1, -1, -1)) & 1
p_x = probs[torch.arange(n), states].prod(dim=1)
inside = (mcmc.counts(mcmc.membership(mdl, states)) >= 1).all(dim=1)
pi = p_x * inside / (p_x * inside).sum()
closed_exact = (pi[:, None] * (states == 0)).sum(dim=0)

g = torch.Generator().manual_seed(0)
t0 = time.perf_counter()
full = mcmc.run(mdl, N_CHAIN, N_BURN + N_KEEP, 0, generator=g, n_partner=N_PARTNER, rho=mcmc.partner_rho(mdl, GAMMA))
wall = time.perf_counter() - t0
per_sweep = ("x", "t", "c", "ok", "changed", "tuple_ok", "tuple_changed")
tr = {**full, **{k: full[k][N_BURN:] for k in per_sweep}, "n_burn": N_BURN}

closed = (tr["x"] == 0).double().mean(dim=(0, 1))
code = (tr["x"].long() * 2 ** torch.arange(n - 1, -1, -1)).sum(dim=-1).flatten()
freq = torch.bincount(code, minlength=2 ** n).double() / len(code)
mx = mcmc.mixing(tr, 2)
var = mx["mean"][:, 0] * (1 - mx["mean"][:, 0]) > 0
acc, moved = mcmc.tuple_rates(tr)
ms = mcmc.timing(tr)
r = {"title": "toy-ca", "systems": mdl.names, "observed": [OBSERVED[o] for o in mdl.names],
     "n_refs": [len(b.lo) for b in mdl.systems], "n_chain": N_CHAIN, "n_burn": N_BURN, "n_keep": N_KEEP,
     "n_partner": N_PARTNER, "gamma": GAMMA, "ref_name": "exact", "P_E": (p_x * inside).sum().item(),
     "P_init": full["p_event"], "closed_err": (closed - closed_exact).abs().max().item(),
     "tv": 0.5 * (freq - pi).abs().sum().item(),
     "rhat_max": mx["rhat"][var, 0].max().item(), "ess_min": mx["ess"][var, 0].min().item(),
     "acceptance": mcmc.acceptance_rate(tr).tolist(), "label_change": mcmc.label_change_rate(tr).tolist(),
     "tuple_accept": acc, "tuple_moved": moved, "occupied": [int((o > 0).sum()) for o in mcmc.occupancy(tr, mdl)],
     "wall_sec": wall, "ms": ms, "edges": edges, "closed_mcmc": closed.tolist(),
     "closed_ref": closed_exact.tolist(), "closed_ref_se": [0.0] * n}

obs = ", ".join(f"{o} {'failed' if s == 0 else 'survived'}" for o, s in zip(mdl.names, r["observed"]))
log(f"toy-ca, {obs}; boxes {r['n_refs']}; P(E) = {r['P_E']:.5f} (exact)")
log(f"{N_CHAIN} chains, {N_BURN} + {N_KEEP} sweeps, each followed by a tuple jump (m = {N_PARTNER}, "
    f"gamma = {GAMMA:g}, random order), sequential start; {wall:.1f} s on cpu")
log(f"P(e closed | E): max error {r['closed_err']:.4f} against the exact posterior; "
    f"TV over the {2 ** n} states {r['tv']:.4f}")
log(f"R-hat max {r['rhat_max']:.4f}, ESS min {r['ess_min']:.0f}; P(E) from the sequential start / exact "
    f"{r['P_init'] / r['P_E']:.4f}")
log(f"sweep: accept {[round(a, 3) for a in r['acceptance']]}, box change {[round(a, 3) for a in r['label_change']]}; "
    f"tuple jump: accept {acc:.3f}, to a partner {moved:.3f} of accepted; boxes occupied {r['occupied']} of "
    f"{r['n_refs']}")
log("ms per sweep (all chains): " + ", ".join(f"{k} {ms[k]:.2f}" for k in mcmc.STEPS + mcmc.TUPLE_STEPS)
    + f"; median sweep {ms['sweep_median']:.2f}")

(HERE / "results.json").write_text(json.dumps(r, indent=2) + "\n")
torch.save({"t": full["t"].short(), "x": full["x"][:, :N_TRACE], "tuple_changed": full["tuple_changed"][:, :N_TRACE],
            "closed_share": (full["x"] == 0).float().mean(dim=1), "n_burn": N_BURN}, HERE / "traces.pt")
(HERE / "run.log").write_text("\n".join(lines) + "\n")
print(f"wrote run.log, results.json and traces.pt in {HERE}")
