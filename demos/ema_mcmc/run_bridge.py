"""Kernel I with and without the population bridge move (mcmc.bridge) on EMA, from the worst
start: one chain at each failure reference state, so every box starts with equal weight.

For the five origins with the most failure reference states, each configuration runs a Kernel I
step and, if set, a bridge step per sweep, and is checked against the weighted-IS reference
(reference_is.pt). Standard errors come from 20 groups of independent chains. Writes
results_bridge.json ({configuration: {origin: result}}) and run_bridge.log.

    PYTHONPATH=. python demos/ema_mcmc/run_bridge.py
"""
import json
import time
from pathlib import Path

import torch

from rsr import mcmc

CONFIGS = {"Kernel I": None, "I + bridge m=1, N=32": (1, 32), "I + bridge m=8, N=128": (8, 128)}
N_BURN, N_KEEP, N_GROUP, N_TOP = 300, 1000, 20, 5
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
data, ref = torch.load(HERE / "ema.pt"), torch.load(HERE / "reference_is.pt")
probs, edges = data["probs"].to(dev), data["edges"]
n = len(edges)
origins = sorted(data["origins"], key=lambda o: -len(data["refs_lower"][o]))[:N_TOP]


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


def run(refs, bridge_cfg):
    """One chain per reference state; returns running sums over the kept sweeps."""
    C = len(refs)
    g = torch.Generator(device=dev).manual_seed(0)
    x, c = mcmc.init(refs, C)                                              # chain k at box k mod R
    closed_sum = torch.zeros(C, n, device=dev)
    c_sum = torch.zeros(C, dtype=torch.float64, device=dev)
    lag_sum, prev = torch.zeros(n, device=dev), None
    moved_sum = left_sum = 0.0
    sec = {"kernel I": 0.0, "bridge": 0.0}
    for t in range(N_BURN + N_KEEP):
        sync(); t0 = time.perf_counter()
        x_new, _ = mcmc.proposal(x, refs, probs, g)
        x, c, _ = mcmc.accept(x_new, x, c, refs, generator=g)
        sync(); t1 = time.perf_counter()
        sec["kernel I"] += t1 - t0
        if bridge_cfg is not None:
            x, c, moved, left = mcmc.bridge(x, c, refs, probs, n_cand=bridge_cfg[1],
                                            n_partner=bridge_cfg[0], generator=g)
            sync()
            sec["bridge"] += time.perf_counter() - t1
            if t >= N_BURN:
                moved_sum += moved.double().mean().item()
                left_sum += left.double().mean().item()
        if t < N_BURN:
            continue
        closed = x[:, :, 0].float()
        closed_sum += closed
        c_sum += c
        if prev is not None:
            lag_sum += (closed * prev).sum(dim=0)
        prev = closed
    return {"closed_chain": closed_sum / N_KEEP, "c_chain": c_sum / N_KEEP, "lag_sum": lag_sum,
            "moved": moved_sum / N_KEEP, "left": left_sum / N_KEEP, "sec": sec, "C": C}


results, lines = {}, []
def log(s):
    print(s, flush=True)
    lines.append(s)

log(f"one chain per failure reference state, burn-in {N_BURN}, {N_KEEP} kept sweeps, on {dev}")
for name, cfg in CONFIGS.items():
    log(f"\n{name}")
    log("origin   R     P_hat/P_IS  max|closed err|  z2        rho1 max  bridge moved  left box  "
        "ms/sweep  s total")
    results[name] = {}
    for o in origins:
        refs = data["refs_lower"][o].to(dev)
        p_box = (probs[None] * refs).sum(dim=2).prod(dim=1)               # for P_hat only
        t0 = time.perf_counter()
        out = run(refs, cfg)
        sec_total = time.perf_counter() - t0
        C = out["C"]
        groups = torch.arange(C, device=dev) % N_GROUP
        def by_group(v):
            return torch.stack([v[groups == k].mean(dim=0) for k in range(N_GROUP)])
        closed = out["closed_chain"].mean(dim=0).double()
        closed_se = by_group(out["closed_chain"]).double().std(dim=0) / N_GROUP ** 0.5
        ref_closed, ref_se = ref["closed"][o].to(dev).double(), ref["closed_se"][o].to(dev).double()
        se = (closed_se ** 2 + ref_se ** 2).sqrt()
        z = (closed - ref_closed)[se > 0] / se[se > 0]
        var = closed * (1 - closed)
        rho1 = torch.where(var > 1e-12, (out["lag_sum"].double() / (C * (N_KEEP - 1)) - closed ** 2)
                           / var.clamp(min=1e-12), torch.zeros_like(var))
        P_hat = (p_box.sum() / out["c_chain"].mean()).item()
        ms = {k: 1e3 * v / (N_BURN + N_KEEP) for k, v in out["sec"].items()}
        r = {"n_refs": len(refs), "n_chain": C, "P_IS": ref["P_union"][o], "P_hat": P_hat,
             "closed_err": (closed - ref_closed).abs().max().item(),
             "z2": (z ** 2).mean().item() if len(z) else 0.0, "rho1_max": rho1.max().item(),
             "bridge_moved": out["moved"], "bridge_left": out["left"], "ms_per_sweep": ms,
             "sec_total": sec_total, "closed_mcmc": closed.tolist(), "closed_is": ref_closed.tolist()}
        results[name][o] = r
        log(f"{o:6s}  {len(refs):4d}  {P_hat / r['P_IS']:7.4f}     {r['closed_err']:.4f}           "
            f"{r['z2']:9.2f}  {r['rho1_max']:.3f}     {r['bridge_moved']:.3f}         "
            f"{r['bridge_left']:.3f}     {sum(ms.values()):7.1f}   {sec_total:6.1f}")

(HERE / "results_bridge.json").write_text(json.dumps(results, indent=2) + "\n")
(HERE / "run_bridge.log").write_text("\n".join(lines) + "\n")
log(f"\nwrote {HERE / 'results_bridge.json'} and run_bridge.log")
