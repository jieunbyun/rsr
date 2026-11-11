"""rsr-mcmc on bn-on-demand's EMA highways (129 edges), each of the 72 origins observed failed.

Chains run over each origin's failure reference states with the exact edge probabilities
(ema.pt) and are checked against the weighted-IS reference (reference_is.pt). Samples are not
stored: the chains update running sums, except the first two chains, whose traces are kept for
plotting. Standard errors come from 20 groups of independent chains.
    P_hat     = sum_r P(L^r) / mean(c)                       P(failure union) from the counts
    closed    = P(e closed | failure union) per edge
    z2        = mean over edges of ((MCMC - IS) / combined SE)^2: about 1 if they agree
    rho1      = lag-1 autocorrelation per edge, pooled over chains; ESS = draws (1 - rho1) / (1 + rho1)

Chains start (--init) at boxes drawn with probability P(L^r) (weighted, 1000 chains), or
--per-box chains at each failure reference state (per-box, R x per-box chains).

    PYTHONPATH=. python demos/ema_mcmc/run_ema.py [--init per-box --per-box 10] [--top 5]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from rsr import mcmc

parser = argparse.ArgumentParser()
parser.add_argument("--init", choices=["weighted", "per-box"], default="weighted")
parser.add_argument("--per-box", type=int, default=10)
parser.add_argument("--top", type=int, default=5, help="origins with the most failure reference states; 0 = all")
args = parser.parse_args()
N_CHAIN, N_BURN, N_KEEP, N_GROUP, N_TRACE = 1000, 500, 2000, 20, 2
SUFFIX = "" if args.init == "weighted" else f"_perbox{args.per_box}"
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
data, ref = torch.load(HERE / "ema.pt"), torch.load(HERE / "reference_is.pt")
probs, edges = data["probs"].to(dev), data["edges"]
n = len(edges)


def timed(sec, name, f, *args, **kwargs):
    torch.cuda.synchronize() if dev == "cuda" else None
    t0 = time.perf_counter()
    out = f(*args, **kwargs)
    torch.cuda.synchronize() if dev == "cuda" else None
    sec[name] += time.perf_counter() - t0
    return out


def run_origin(refs, n_chain):
    g = torch.Generator(device=dev).manual_seed(0)
    sec = {"proposal": 0.0, "accept": 0.0}
    if args.init == "weighted":
        x, c = mcmc.init(refs, n_chain, probs=probs, generator=g)          # boxes drawn by P(L^r)
    else:
        x, c = mcmc.init(refs, n_chain)                                    # chain k at box k mod R
    closed_sum = torch.zeros(n_chain, n, device=dev)                       # per chain
    c_sum = torch.zeros(n_chain, dtype=torch.float64, device=dev)
    ok_sum = torch.zeros(n_chain, device=dev)
    lag_sum, prev = torch.zeros(n, device=dev), None
    visits = torch.zeros(len(refs), device=dev)
    trace_x, trace_g = [], []
    for t in range(N_BURN + N_KEEP):
        x_new, gamma = timed(sec, "proposal", mcmc.proposal, x, refs, probs, g)
        x, c, ok = timed(sec, "accept", mcmc.accept, x_new, x, c, refs, generator=g)
        if t < N_BURN:
            continue
        closed = x[:, :, 0].float()                                        # (C, n), 1 = closed
        closed_sum += closed
        c_sum += c
        ok_sum += ok
        if prev is not None:
            lag_sum += (closed * prev).sum(dim=0)
        prev = closed
        visits += torch.bincount(gamma, minlength=len(refs))
        trace_x.append(x[:N_TRACE, :, 1].to(torch.int8).cpu())
        trace_g.append(gamma[:N_TRACE].to(torch.int16).cpu())
    return {"closed_chain": closed_sum / N_KEEP, "c_chain": c_sum / N_KEEP, "ok": ok_sum.mean() / N_KEEP,
            "lag_sum": lag_sum, "visits": visits, "sec": sec,
            "trace_x": torch.stack(trace_x), "trace_g": torch.stack(trace_g)}


results, traces = {}, {}
chains = f"{N_CHAIN} chains" if args.init == "weighted" else f"{args.per_box} chains per failure reference state"
print(f"{chains} ({args.init} start), burn-in {N_BURN}, {N_KEEP} kept steps, on {dev}\n")
print("origin   R     chains  P_IS        P_hat/P_IS  z(P)   max|closed err|  z2    accept  rho1 max  "
      "ESS min  ms/step  s total")
order = sorted(data["origins"], key=lambda o: -len(data["refs_lower"][o]))[:args.top or None]
order = sorted(order, key=lambda o: -ref["P_union"][o])
t_all = time.perf_counter()
for o in order:
    refs = data["refs_lower"][o].to(dev)
    p_box = (probs[None] * refs).sum(dim=2).prod(dim=1)
    n_chain = N_CHAIN if args.init == "weighted" else args.per_box * len(refs)
    t_origin = time.perf_counter()
    out = run_origin(refs, n_chain)
    sec_total = time.perf_counter() - t_origin

    groups = torch.arange(n_chain, device=dev) % N_GROUP
    def by_group(v):                                                       # (C, ...) -> (G, ...)
        return torch.stack([v[groups == k].mean(dim=0) for k in range(N_GROUP)])
    closed = out["closed_chain"].mean(dim=0).double()
    closed_se = by_group(out["closed_chain"]).double().std(dim=0) / N_GROUP ** 0.5
    P_g = p_box.sum() / by_group(out["c_chain"])
    P_hat, P_se = (p_box.sum() / out["c_chain"].mean()).item(), (P_g.std() / N_GROUP ** 0.5).item()

    ref_closed, ref_se = ref["closed"][o].to(dev).double(), ref["closed_se"][o].to(dev).double()
    se = (closed_se ** 2 + ref_se ** 2).sqrt()
    z = (closed - ref_closed)[se > 0] / se[se > 0]
    m = closed                                                             # pooled mean per edge
    draws = n_chain * (N_KEEP - 1)
    var = m * (1 - m)
    rho1 = torch.where(var > 1e-12, (out["lag_sum"].double() / draws - m ** 2) / var.clamp(min=1e-12),
                       torch.zeros_like(var))
    ess = (n_chain * N_KEEP * (1 - rho1) / (1 + rho1)).clamp(max=n_chain * N_KEEP)
    P_ref, P_ref_se = ref["P_union"][o], ref["P_union_se"][o]
    r = {
        "n_refs": len(refs), "n_chain": n_chain, "sec_total": sec_total, "P_IS": P_ref, "P_IS_se": P_ref_se, "P_hat": P_hat, "P_hat_se": P_se,
        # both SEs are 0 when every state lies in one box (c = 1): both estimates are then sum_r P(L^r)
        "z_P": (P_hat - P_ref) / se_P if (se_P := (P_se ** 2 + P_ref_se ** 2) ** 0.5) > 0 else 0.0,
        "closed_err": (closed - ref_closed).abs().max().item(),
        "z2": (z ** 2).mean().item() if len(z) else 0.0,
        "acceptance": out["ok"].item(), "rho1_max": rho1.max().item(),
        "ess_min": ess[var > 1e-12].min().item() if (var > 1e-12).any() else float(n_chain * N_KEEP),
        "boxes_visited": int((out["visits"] > 0).sum()),
        "ms_per_step": {k: 1e3 * v / (N_BURN + N_KEEP) for k, v in out["sec"].items()},
        "closed_mcmc": closed.tolist(), "closed_is": ref_closed.tolist(),
    }
    results[o] = r
    traces[o] = {"x": out["trace_x"], "gamma": out["trace_g"], "visits": out["visits"].cpu(),
                 "closed": closed.cpu()}
    ms = sum(r["ms_per_step"].values())
    print(f"{o:6s}  {r['n_refs']:4d}  {n_chain:6d}  {P_ref:.3e}   {P_hat / P_ref:7.4f}   {r['z_P']:+5.1f}   "
          f"{r['closed_err']:.4f}           {r['z2']:5.2f}  {r['acceptance']:.3f}   {r['rho1_max']:.3f}    "
          f"{r['ess_min']:8.0f}  {ms:6.2f}  {sec_total:7.1f}", flush=True)

(HERE / f"results_ema{SUFFIX}.json").write_text(json.dumps(results, indent=2) + "\n")
torch.save({"edges": edges, "traces": traces, "results": results}, HERE / f"traces_ema{SUFFIX}.pt")
print(f"\n{len(order)} origins in {time.perf_counter() - t_all:.0f} s; wrote results_ema{SUFFIX}.json, traces_ema{SUFFIX}.pt")
