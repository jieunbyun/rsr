"""Scaling of rsr.mcmc (sweep + tuple jump) with the number of observed systems, on
bn-on-demand's EMA highways (129 binary edges).

The systems are the N_MAX origins with the most failure reference states among those whose
failure boxes have total probability at least P_MIN (rarer origins make the joint observation
too unlikely for the reference to sample), in that order. For J = 1, ..., N_MAX the first J are
observed:
    fail   all J failed (failure boxes, refs_lower)
    mixed  the first ceil(J / 2) failed and the rest surviving (survival boxes, refs_upper);
           from J = 2, since for J = 1 it is the fail case
The event is certified: a system "fails" when the state lies in one of its failure boxes and
"survives" when it lies in one of its survival boxes (the stored reference states are not
complete, so this is the event both the chains and the reference condition on).

Reference: weighted importance sampling from the failed system whose failure boxes have the
smallest total probability (the anchor a): box r with probability P(L^{a,r}), x ~ p(. | L^{a,r}),
weight 1(x in E) / c_a(x); P(E) = sum_r P(L^{a,r}) * mean weight. Its effective sample size
(ess_ref) is reported, and cached in reference_scale.pt.

MCMC: N_CHAIN chains from the sequential start, each sweep followed by a tuple jump with
N_PARTNER partners from rho of Eq. (rho), gamma = 1, random order. Writes results_scale.json,
traces_scale/<series>_<J>.pt (labels of every chain and the share of chains with each edge
closed per sweep, burn-in included) and run_scale.log.

    PYTHONPATH=. python demos/ema_mcmc/run_scale.py            # full run
    PYTHONPATH=. python demos/ema_mcmc/run_scale.py --smoke    # short check, writes *_smoke files
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from rsr import mcmc

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--cases", nargs="*", help="only these, e.g. fail/3 mixed/4")
args = parser.parse_args()

N_MAX, N_PARTNER, GAMMA, N_GROUP, P_MIN = 10, 8, 1.0, 20, 1e-5
if args.smoke:
    N_CHAIN, N_BURN, N_KEEP, N_REF = 200, 50, 100, 400_000
else:
    N_CHAIN, N_BURN, N_KEEP, N_REF = 1000, 500, 1000, 4_000_000
REF_BATCH = 50_000
SUFFIX = "_smoke" if args.smoke else ""
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.load(HERE / "ema.pt")
probs = data["probs"].to(dev)
n, M = probs.shape
lines = []


def fail_prob(o):
    """sum_r P(L^r) over the failure boxes of origin o (at least P(failure certified))."""
    mdl = mcmc.model([data["refs_lower"][o]], probs)
    return mcmc.label_logprob(mdl, 0, torch.zeros(1, n, dtype=torch.long, device=dev),
                              torch.full((1, n), M - 1, device=dev))[0].exp().sum().item()


P_FAIL = {o: fail_prob(o) for o in data["origins"]}
ORIGINS = sorted((o for o in data["origins"] if P_FAIL[o] >= P_MIN), key=lambda o: -len(data["refs_lower"][o]))[:N_MAX]


def log(s):
    print(s, flush=True)
    lines.append(s)


def cases():
    out = []
    for J in range(1, N_MAX + 1):
        out.append(("fail", J, ORIGINS[:J], ["fail"] * J))
    for J in range(2, N_MAX + 1):
        k = math.ceil(J / 2)
        out.append(("mixed", J, ORIGINS[:J], ["fail"] * k + ["survive"] * (J - k)))
    return [c for c in out if not args.cases or f"{c[0]}/{c[1]}" in args.cases]


def build(systems, states):
    refs = [data["refs_lower" if s == "fail" else "refs_upper"][o] for o, s in zip(systems, states)]
    return mcmc.model(refs, probs)


def box_logprob(mdl, j):
    return mcmc.label_logprob(mdl, j, torch.zeros(1, n, dtype=torch.long, device=dev),
                              torch.full((1, n), M - 1, device=dev))[0]


def reference(name, mdl, states):
    """Weighted IS from the anchor's failure boxes (see the module docstring)."""
    cache_path = HERE / f"reference_scale{SUFFIX}.pt"
    cache = torch.load(cache_path) if cache_path.exists() else {}
    if name in cache:
        return cache[name]
    t0 = time.perf_counter()
    lps = {j: box_logprob(mdl, j) for j, s in enumerate(states) if s == "fail"}
    a = min(lps, key=lambda j: torch.logsumexp(lps[j], 0).item())
    lp, bx = lps[a], mdl.systems[a]
    g = torch.Generator(device=dev).manual_seed(1)
    num, den, w2 = [], [], 0.0
    for _ in range(N_REF // REF_BATCH):
        r = torch.multinomial((lp - lp.max()).exp(), REF_BATCH, replacement=True, generator=g)
        x = mcmc.draw_state(mdl, bx.lo[r], bx.hi[r], g)
        c = mcmc.counts(mcmc.membership(mdl, x)).double()
        w = (c >= 1).all(dim=1) / c[:, a]
        num.append((w[:, None] * (x == 0)).sum(dim=0))
        den.append(w.sum())
        w2 += (w ** 2).sum().item()
    num, den = torch.stack(num), torch.stack(den)
    k = len(den) // N_GROUP
    closed_b = num[:k * N_GROUP].view(N_GROUP, k, n).sum(1) / den[:k * N_GROUP].view(N_GROUP, k).sum(1)[:, None].clamp(min=1e-300)
    out = {"closed": (num.sum(0) / den.sum()).cpu(), "closed_se": (closed_b.std(0) / N_GROUP ** 0.5).cpu(),
           "P": (torch.logsumexp(lp, 0).exp() * den.sum() / N_REF).item(), "n_draw": N_REF,
           "ess": (den.sum().item() ** 2 / w2) if w2 > 0 else 0.0, "anchor": a, "sec": time.perf_counter() - t0}
    cache[name] = out
    torch.save(cache, cache_path)
    return out


def evaluate(series, J, systems, states):
    name = f"{series}/{J}"
    mdl = build(systems, states)
    ref = reference(name, mdl, states)
    g = torch.Generator(device=dev).manual_seed(0)
    t0 = time.perf_counter()
    full = mcmc.run(mdl, N_CHAIN, N_BURN + N_KEEP, 0, generator=g, n_partner=N_PARTNER,
                    rho=mcmc.partner_rho(mdl, GAMMA))
    wall = time.perf_counter() - t0
    tr = {**full, **{k: full[k][N_BURN:] for k in ("x", "t", "c", "ok", "changed", "tuple_ok", "tuple_changed")},
          "n_burn": N_BURN}
    closed_chain = (tr["x"] == 0).double().mean(dim=0)
    closed = closed_chain.mean(dim=0)
    groups = torch.arange(N_CHAIN, device=dev) % N_GROUP
    closed_se = torch.stack([closed_chain[groups == k].mean(0) for k in range(N_GROUP)]).std(0) / N_GROUP ** 0.5
    ref_closed, ref_se = ref["closed"].to(dev).double(), ref["closed_se"].to(dev).double()
    err = (closed - ref_closed).abs()
    se = (closed_se ** 2 + ref_se ** 2).sqrt()
    z = (closed - ref_closed)[se > 0] / se[se > 0]
    mx = mcmc.mixing(tr, 2)
    var = mx["mean"][:, 0] * (1 - mx["mean"][:, 0]) > 0
    acc, moved = mcmc.tuple_rates(tr)
    ms = mcmc.timing(tr)
    r = {"series": series, "J": J, "systems": systems, "states": states, "n_refs": [len(b.lo) for b in mdl.systems],
         "n_chain": N_CHAIN, "n_burn": N_BURN, "n_keep": N_KEEP, "n_partner": N_PARTNER, "gamma": GAMMA,
         "P_ref": ref["P"], "ess_ref": ref["ess"], "anchor": systems[ref["anchor"]], "P_init": full["p_event"],
         "closed_err": err.max().item(), "n_off": int((err > 0.05).sum()), "z2": (z ** 2).mean().item() if len(z) else 0.0,
         "rhat_max": mx["rhat"][var, 0].max().item() if var.any() else 1.0,
         "ess_min": mx["ess"][var, 0].min().item() if var.any() else float(N_CHAIN * N_KEEP),
         "acceptance": mcmc.acceptance_rate(tr).tolist(), "label_change": mcmc.label_change_rate(tr).tolist(),
         "tuple_accept": acc, "tuple_moved": moved,
         "occupied": [int((o > 0).sum()) for o in mcmc.occupancy(tr, mdl)],
         "wall_sec": wall, "ms": ms, "closed_mcmc": closed.tolist(), "closed_ref": ref_closed.tolist()}
    trace = {"t": full["t"].short().cpu(), "closed_share": (full["x"] == 0).float().mean(dim=1).cpu(),
             "tuple_changed": full["tuple_changed"].cpu(), "n_burn": N_BURN}
    p_init = f"{full['p_event'] / ref['P']:.3f}" if ref["P"] > 0 else "  -  "
    log(f"{series:5s} {J:2d}  {ref['P']:.2e}  {ref['ess']:9.0f}  {p_init:>7s}  {r['closed_err']:.4f}  {r['n_off']:3d}  "
        f"{r['z2']:7.1f}  {r['rhat_max']:6.3f}  {r['ess_min']:8.0f}  {acc:.3f}  {moved:.4f}  "
        f"{sum(r['occupied']):5d}  {ms['sweep_median']:7.1f}  {ms['partners']:7.1f}  {wall:6.0f}")
    return r, trace


log(f"EMA, {N_CHAIN} chains, {N_BURN} + {N_KEEP} sweeps, tuple jump m = {N_PARTNER}, gamma = {GAMMA}, random order; "
    f"reference: {N_REF:,} weighted-IS draws; on {dev}")
log(f"systems in order (failure boxes' total probability >= {P_MIN:g}): "
    + ", ".join(f"{o} ({len(data['refs_lower'][o])}, {P_FAIL[o]:.1e})" for o in ORIGINS)
    + "\nmixed: the first ceil(J/2) failed, the rest surviving\n")
log("case   J  P_ref     ESS_ref    P_init/P  err     >.05  z2       R-hat   ESS min   t-acc  t-moved occ    "
    "ms/sweep partners  sec")
results = {}
for series, J, systems, states in cases():
    r, tr = evaluate(series, J, systems, states)
    results[f"{series}/{J}"] = r
    (HERE / f"results_scale{SUFFIX}.json").write_text(json.dumps(results, indent=2) + "\n")
    (HERE / f"traces_scale{SUFFIX}").mkdir(exist_ok=True)
    torch.save({**tr, "result": r, "edges": data["edges"]}, HERE / f"traces_scale{SUFFIX}" / f"{series}_{J}.pt")
    (HERE / f"run_scale{SUFFIX}.log").write_text("\n".join(lines) + "\n")
print(f"\nwrote results_scale{SUFFIX}.json, traces_scale{SUFFIX}/ and run_scale{SUFFIX}.log in {HERE}")
