"""Crude Monte Carlo (MCS) check of run.py: draw edge states from the prior, keep the draws in
which both observed systems fail (inside a failure box of n57 and one of n58), and compare each
edge's P(closed | E) among them with the MCMC and weighted-IS posteriors in results.json.

The failure boxes are cut sets (edges required closed, the rest free), so a draw is in a box iff
it closes every edge of the box; this is checked by a matmul. Draws with fewer closed edges than
the smallest box of either system are skipped before the check. P(E) is about 1.5e-7, so N_MCS
draws give about 1.5e-7 * N_MCS hits.

Writes mcs.log, mcs.json and mcs.png; a later run with the same N_MCS reuses the counts in
mcs.json and only redoes the comparison. Run run.py first.

    PYTHONPATH=. python demos/ema_mcmc/mcs.py
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

N_MCS, BATCH, SEED = 30_000_000_000, 2_000_000, 1
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.load(HERE / "ema_n57_n58.pt")
p_closed = data["probs"][:, 0].to(dev).float()
n = len(p_closed)
dt = torch.float16 if dev == "cuda" else torch.float32
cuts = [(~data["refs_lower"][o][:, :, 1]).to(dev) for o in data["origins"]]   # (R, n): edges required closed
A = [c.to(dt).T for c in cuts]
size = [c.sum(1).to(dt) for c in cuts]
k_min = max(int(c.sum(1).min()) for c in cuts)    # both must fail, so each draw needs at least this many closures


def fails(x, j):
    return ((x.to(dt) @ A[j]) == size[j]).any(dim=1)


def sample():
    g = torch.Generator(device=dev).manual_seed(SEED)
    hits, closed_sum, t0 = 0, torch.zeros(n, dtype=torch.float64, device=dev), time.perf_counter()
    for b in range(N_MCS // BATCH):
        x = torch.rand(BATCH, n, device=dev, generator=g) < p_closed
        x = x[x.sum(1) >= k_min]
        for j in range(len(A)):
            x = x[fails(x, j)]
        hits += len(x)
        closed_sum += x.sum(0)
        if (b + 1) % 1000 == 0:
            print(f"{(b + 1) * BATCH:.2e} draws, {hits} hits, {time.perf_counter() - t0:.0f} s", flush=True)
    return hits, closed_sum.cpu(), time.perf_counter() - t0


path = HERE / "mcs.json"     # the draws are reused when mcs.json holds as many, so the comparison can be redone
if path.exists() and (old := json.loads(path.read_text()))["n_draw"] == N_MCS and "closed_count" in old:
    hits, closed_sum, sec = old["hits"], torch.tensor(old["closed_count"], dtype=torch.float64), old["sec"]
else:
    hits, closed_sum, sec = sample()

closed = closed_sum / hits
c_se = (closed_sum + 0.5) / (hits + 1)           # so that edges never or always closed get SE > 0
se = (c_se * (1 - c_se) / hits).sqrt()
r = json.loads((HERE / "results.json").read_text())
mcmc_c, ref_c = torch.tensor(r["closed_mcmc"]), torch.tensor(r["closed_ref"])
P = hits / N_MCS
lines = [f"MCS: {N_MCS:.1e} prior draws on {dev}, {hits} in E (n57 and n58 failed); P(E) = {P:.3e} "
         f"+/- {(P * (1 - P) / N_MCS) ** 0.5:.1e} (IS reference {r['P_E']:.3e}); {sec:.0f} s"]
for name, c in (("MCMC", mcmc_c), ("weighted IS", ref_c)):
    e = (c - closed).abs()
    lines.append(f"{name:>11} vs MCS: max |error| {e.max():.4f}, mean {e.mean():.4f}; "
                 f"{int((e > 3 * se).sum())}/{n} edges off by > 3 MCS standard errors")
print("\n".join(lines))
(HERE / "mcs.log").write_text("\n".join(lines) + "\n")
(HERE / "mcs.json").write_text(json.dumps({"n_draw": N_MCS, "hits": hits, "P_E": P, "sec": sec, "edges": r["edges"],
                                           "closed_count": closed_sum.tolist(), "closed_mcs": closed.tolist(), "closed_mcs_se": se.tolist()}, indent=2) + "\n")

fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
for ax, (name, c) in zip(axs, (("MCMC", mcmc_c), ("weighted IS", ref_c))):
    ax.errorbar(closed, c, xerr=2 * se, fmt="o", ms=3, elinewidth=0.6, alpha=0.8)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="P(closed | E), MCS (±2 SE)", ylabel=f"P(closed | E), {name}",
           title=f"{name} vs MCS: max |error| {(c - closed).abs().max():.3f}")
fig.suptitle(f"EMA, n57 and n58 failed; MCS {hits} hits in {N_MCS:.0e} draws")
fig.tight_layout()
fig.savefig(HERE / "mcs.png", dpi=150)
print(f"wrote mcs.log, mcs.json and mcs.png in {HERE}")
