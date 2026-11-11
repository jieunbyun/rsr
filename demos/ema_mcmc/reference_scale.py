"""Reference posteriors for run_scale.py by weighted importance sampling from the intersection of
two observed failures (the anchor pair), which reaches much rarer joint observations than
sampling from one system's boxes.

Proposal: a box pair (r1, r2) of the anchors with probability P(L^{a1,r1} n L^{a2,r2}) / Z12,
then x ~ p(. | L^{a1,r1} n L^{a2,r2}). Its density is p(x) c_a1(x) c_a2(x) / Z12, so the weight
1(x in E) / (c_a1(x) c_a2(x)) gives P(E) = Z12 * mean weight and the self-normalised posterior.
The anchors are the pair, among the three failed systems with the smallest total box
probability, with the smallest Z12. Cases with one failed system keep the one-system anchor.
Reports the effective sample size of the weights, and writes reference_scale_pair.pt; then
recomputes the errors of results_scale.json against it (results_scale_pair.json).

    PYTHONPATH=. python demos/ema_mcmc/reference_scale.py [--n-draw 4000000] [--cases fail/4 ...]
"""
import argparse
import itertools
import json
import math
import time
from pathlib import Path

import torch

from rsr import mcmc

parser = argparse.ArgumentParser()
parser.add_argument("--n-draw", type=int, default=4_000_000)
parser.add_argument("--cases", nargs="*")
parser.add_argument("--results", default="results_scale.json")
args = parser.parse_args()

N_GROUP, BATCH, ROW_CHUNK = 20, 100_000, 32
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.load(HERE / "ema.pt")
probs = data["probs"].to(dev)
n, M = probs.shape
res_path = HERE / args.results
results = json.loads(res_path.read_text()) if res_path.exists() else {}


def build(systems, states):
    refs = [data["refs_lower" if s == "fail" else "refs_upper"][o] for o, s in zip(systems, states)]
    return mcmc.model(refs, probs)


def pair_logprob(mdl, j1, j2):
    """log P(L^{j1,r1} n L^{j2,r2}), (R1, R2)."""
    b1, b2 = mdl.systems[j1], mdl.systems[j2]
    comp = torch.arange(n, device=dev)
    out = torch.empty(len(b1.lo), len(b2.lo), dtype=torch.float64, device=dev)
    for s in range(0, len(b1.lo), ROW_CHUNK):
        lo = torch.maximum(b1.lo[s:s + ROW_CHUNK, None], b2.lo[None])            # (k, R2, n)
        hi = torch.minimum(b1.hi[s:s + ROW_CHUNK, None], b2.hi[None])
        empty = (lo > hi).any(dim=-1)
        lp = mcmc._mass(mdl.cdf, comp.expand_as(lo), lo, torch.maximum(lo, hi)).log().sum(dim=-1)
        out[s:s + ROW_CHUNK] = torch.where(empty, -torch.inf, lp)
    return out


def single_logprob(mdl, j):
    return mcmc.label_logprob(mdl, j, torch.zeros(1, n, dtype=torch.long, device=dev),
                              torch.full((1, n), M - 1, device=dev))[0]


def reference(mdl, states, g):
    t0 = time.perf_counter()
    fails = [j for j, s in enumerate(states) if s == "fail"]
    single = {j: single_logprob(mdl, j) for j in fails}
    rare = sorted(fails, key=lambda j: torch.logsumexp(single[j], 0).item())[:3]
    if len(fails) >= 2:
        pairs = {(a, b): pair_logprob(mdl, a, b) for a, b in itertools.combinations(rare, 2)}
        anchor = min(pairs, key=lambda k: torch.logsumexp(pairs[k].flatten(), 0).item())
        lp = pairs[anchor].flatten()
        R2 = len(mdl.systems[anchor[1]].lo)
    else:
        anchor = (rare[0],)
        lp = single[rare[0]]
    log_Z = torch.logsumexp(lp, 0).item()
    w_cat = (lp - lp.max()).exp()
    num, den, w2 = [], [], 0.0
    for _ in range(args.n_draw // BATCH):
        idx = torch.multinomial(w_cat, BATCH, replacement=True, generator=g)
        if len(anchor) == 2:
            r1, r2 = idx // R2, idx % R2
            b1, b2 = mdl.systems[anchor[0]], mdl.systems[anchor[1]]
            lo, hi = torch.maximum(b1.lo[r1], b2.lo[r2]), torch.minimum(b1.hi[r1], b2.hi[r2])
        else:
            b = mdl.systems[anchor[0]]
            lo, hi = b.lo[idx], b.hi[idx]
        x = mcmc.draw_state(mdl, lo, hi, g)
        c = mcmc.counts(mcmc.membership(mdl, x)).double()
        w = (c >= 1).all(dim=1) / c[:, list(anchor)].prod(dim=1)
        num.append((w[:, None] * (x == 0)).sum(dim=0))
        den.append(w.sum())
        w2 += (w ** 2).sum().item()
    num, den = torch.stack(num), torch.stack(den)
    k = len(den) // N_GROUP
    closed_b = (num[:k * N_GROUP].view(N_GROUP, k, n).sum(1)
                / den[:k * N_GROUP].view(N_GROUP, k).sum(1)[:, None].clamp(min=1e-300))
    D = den.sum().item()
    return {"closed": (num.sum(0) / max(D, 1e-300)).cpu(), "closed_se": (closed_b.std(0) / N_GROUP ** 0.5).cpu(),
            "P": math.exp(log_Z) * D / args.n_draw, "ess": D ** 2 / w2 if w2 > 0 else 0.0,
            "anchor": list(anchor), "log_Z": log_Z, "n_draw": args.n_draw, "sec": time.perf_counter() - t0}


cases = sorted(results, key=lambda k: (k.split("/")[0], int(k.split("/")[1])))
if args.cases:
    cases = [c for c in cases if c in args.cases]
cache_path = HERE / "reference_scale_pair.pt"
cache = torch.load(cache_path) if cache_path.exists() else {}
print(f"{args.n_draw:,} draws per case, anchored on the intersection of two failed systems; on {dev}")
print("case       anchor           P_ref      ESS       one-system ESS  err (one-system ref)  err (pair ref)  >.05  sec")
out = {}
for name in cases:
    r = results[name]
    mdl = build(r["systems"], r["states"])
    if name not in cache or cache[name]["n_draw"] != args.n_draw:
        cache[name] = reference(mdl, r["states"], torch.Generator(device=dev).manual_seed(2))
        torch.save(cache, cache_path)
    ref = cache[name]
    mc = torch.tensor(r["closed_mcmc"], dtype=torch.float64)
    err = (mc - ref["closed"]).abs() if ref["ess"] > 0 else torch.full_like(mc, float("nan"))
    out[name] = {**r, "P_ref": ref["P"], "ess_ref": ref["ess"], "anchor": [r["systems"][j] for j in ref["anchor"]],
                 "closed_ref": ref["closed"].tolist(), "closed_err": err.max().item(), "n_off": int((err > 0.05).sum()),
                 "closed_err_one_system": r["closed_err"], "ess_ref_one_system": r["ess_ref"]}
    print(f"{name:10s} {'+'.join(out[name]['anchor']):16s} {ref['P']:.2e}  {ref['ess']:9.0f}  {r['ess_ref']:12.0f}  "
          f"{r['closed_err']:18.4f}  {err.max().item():14.4f}  {int((err > 0.05).sum()):4d}  {ref['sec']:4.0f}", flush=True)
(HERE / res_path.name.replace(".json", "_pair.json")).write_text(json.dumps(out, indent=2) + "\n")
print("wrote", res_path.name.replace(".json", "_pair.json"))
