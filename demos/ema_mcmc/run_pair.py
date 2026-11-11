"""Two EMA systems observed failed together (n57 and n58): step_multi with and without the
multi-system bridge move, from the worst start.

The reference posterior given both failures is weighted IS on n58 (box r with probability
P(L^r), x ~ p(. | L^r), weight 1 / c_58(x)) keeping only draws in n57's failure union; about a
quarter of the draws qualify. Chains start at init_multi's default, box (k mod R_s) of each
system, so the starting tuples ignore which pairs of boxes are likely together. Each sweep is a
step_multi; a bridge step over box tuples follows when configured, then the labels are refreshed.
Standard errors come from 20 groups of independent chains. Writes results_pair.json and
run_pair.log.

    PYTHONPATH=. python demos/ema_mcmc/run_pair.py
"""
import json
import time
from pathlib import Path

import torch

from rsr import mcmc

SYSTEMS = ("n57", "n58")
CONFIGS = {"step_multi": None,
           "+ bridge m=8, N=128, uniform partners": (8, 128, None),
           "+ bridge m=8, N=128, overlap partners (beta=0.1)": (8, 128, 0.1)}
N_BURN, N_KEEP, N_GROUP, N_REF, REF_BATCH = 300, 1000, 20, 4_000_000, 100_000
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.load(HERE / "ema.pt")
probs, edges = data["probs"].to(dev), data["edges"]
n = len(edges)
refs_list = [data["refs_lower"][o].to(dev) for o in SYSTEMS]
lines = []


def log(s):
    print(s, flush=True)
    lines.append(s)


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


# --- reference: weighted IS on n58, restricted to n57's failure union ---
def reference():
    g = torch.Generator(device=dev).manual_seed(1)
    a, b = refs_list[1], refs_list[0]                                      # draw from n58, keep n57
    p_box = (probs[None] * a).sum(dim=2).prod(dim=1)
    num, den, keep = [], [], 0
    for _ in range(N_REF // REF_BATCH):
        r = torch.multinomial(p_box, REF_BATCH, replacement=True, generator=g)
        x = mcmc._draw(probs[None] * a[r], g)
        w = (1.0 / mcmc._membership(x, a).sum(dim=1).double()) * mcmc._membership(x, b).any(dim=1)
        num.append((w[:, None] * x[:, :, 0].double()).sum(dim=0))
        den.append(w.sum())
        keep += int((w > 0).sum())
    num, den = torch.stack(num), torch.stack(den)
    closed_b = num / den[:, None]
    return {"closed": num.sum(0) / den.sum(), "closed_se": closed_b.std(0) / len(den) ** 0.5,
            "P_both": (p_box.sum() * den.sum() / N_REF).item(), "kept": keep}


def run(cfg):
    C = max(len(r) for r in refs_list)
    g = torch.Generator(device=dev).manual_seed(0)
    x, gamma, c = mcmc.init_multi(refs_list, C)
    closed_sum = torch.zeros(C, n, device=dev)
    lag_sum, prev = torch.zeros(n, device=dev), None
    moved_sum = left_sum = 0.0
    sec = {"step_multi": 0.0, "bridge": 0.0}
    for t in range(N_BURN + N_KEEP):
        sync(); t0 = time.perf_counter()
        x, gamma, c, _ = mcmc.step_multi(x, gamma, c, refs_list, probs, g)
        sync(); t1 = time.perf_counter()
        sec["step_multi"] += t1 - t0
        if cfg is not None:
            m, N, beta = cfg
            x, c, moved, left = mcmc.bridge(x, c, refs_list, probs, n_cand=N, n_partner=m,
                                            partner_beta=beta, generator=g)
            gamma = mcmc.labels(x, refs_list, g)
            sync()
            sec["bridge"] += time.perf_counter() - t1
            if t >= N_BURN:
                moved_sum += moved.double().mean().item()
                left_sum += left.double().mean().item()
        if t < N_BURN:
            continue
        closed = x[:, :, 0].float()
        closed_sum += closed
        if prev is not None:
            lag_sum += (closed * prev).sum(dim=0)
        prev = closed
    return {"closed_chain": closed_sum / N_KEEP, "lag_sum": lag_sum, "C": C, "sec": sec,
            "moved": moved_sum / N_KEEP, "left": left_sum / N_KEEP}


t0 = time.perf_counter()
ref = reference()
log(f"reference: {N_REF:,} weighted-IS draws on n58, {ref['kept']:,} also fail n57; "
    f"P(both fail) = {ref['P_both']:.3e} ({time.perf_counter() - t0:.0f} s)")
log(f"systems {SYSTEMS} with {[len(r) for r in refs_list]} failure reference states; "
    f"burn-in {N_BURN}, {N_KEEP} kept sweeps, on {dev}\n")
log("configuration                                      chains  max|closed err|  z2        rho1 max  "
    "moved  left box  ms/sweep  s total")
results = {"reference": {"systems": list(SYSTEMS), "P_both": ref["P_both"], "closed": ref["closed"].tolist()}}
for name, cfg in CONFIGS.items():
    t0 = time.perf_counter()
    out = run(cfg)
    sec_total = time.perf_counter() - t0
    C = out["C"]
    groups = torch.arange(C, device=dev) % N_GROUP
    closed = out["closed_chain"].mean(dim=0).double()
    closed_se = torch.stack([out["closed_chain"][groups == k].mean(dim=0) for k in range(N_GROUP)]
                            ).double().std(dim=0) / N_GROUP ** 0.5
    se = (closed_se ** 2 + ref["closed_se"] ** 2).sqrt()
    z = (closed - ref["closed"])[se > 0] / se[se > 0]
    var = closed * (1 - closed)
    rho1 = torch.where(var > 1e-12, (out["lag_sum"].double() / (C * (N_KEEP - 1)) - closed ** 2)
                       / var.clamp(min=1e-12), torch.zeros_like(var))
    ms = {k: 1e3 * v / (N_BURN + N_KEEP) for k, v in out["sec"].items()}
    r = {"n_chain": C, "closed_err": (closed - ref["closed"]).abs().max().item(),
         "z2": (z ** 2).mean().item(), "rho1_max": rho1.max().item(), "bridge_moved": out["moved"],
         "bridge_left": out["left"], "ms_per_sweep": ms, "sec_total": sec_total,
         "closed_mcmc": closed.tolist(), "closed_is": ref["closed"].tolist(),
         "P_IS": ref["P_both"], "n_refs": sum(len(r) for r in refs_list)}
    results[name] = r
    log(f"{name:50s} {C:6d}  {r['closed_err']:.4f}           {r['z2']:9.2f}  {r['rho1_max']:.3f}     "
        f"{r['bridge_moved']:.3f}  {r['bridge_left']:.3f}     {sum(ms.values()):7.1f}   {sec_total:6.1f}")

(HERE / "results_pair.json").write_text(json.dumps(results, indent=2) + "\n")
(HERE / "run_pair.log").write_text("\n".join(lines) + "\n")
log(f"\nwrote {HERE / 'results_pair.json'} and run_pair.log")
