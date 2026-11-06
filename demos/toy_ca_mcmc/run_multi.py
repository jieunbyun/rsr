"""rsr-mcmc on toy-ca with several systems observed failed: n4 and n6, then n4, n6 and n7.

Each sweep processes the observed systems one by one (mcmc.step_multi): redraw that system's
box gamma_s, propose inside every system's current box, accept by the product of overlap
counts. The chains target p(x) restricted to the states that fail every observed system's
failure boxes. All 1024 edge configurations are enumerated for the exact answers.

    PYTHONPATH=. python demos/toy_ca_mcmc/run_multi.py      (then plot_traces.py)
"""
import json
import time
from pathlib import Path

import torch

from rsr import mcmc

CASES = [("n4", "n6"), ("n4", "n6", "n7")]
N_CHAIN, N_BURN, N_KEEP = 2, 200, 10_000
HERE = Path(__file__).resolve().parent
data = torch.load(HERE / "toy_ca.pt")
probs, edges = data["probs"], data["edges"]
states = (torch.arange(1024)[:, None] >> torch.arange(9, -1, -1)) & 1   # (1024, 10), as exported
p_x = probs[torch.arange(10), states].prod(dim=1)


def exact(refs_list, systems):
    """pi(x) on the joint union, pi(gamma_s = r) per system, P(e_i closed | obs), P(all fail)."""
    onehot = torch.nn.functional.one_hot(states, 2).int()
    chis = [mcmc._membership(onehot, r) for r in refs_list]                 # (1024, R_s) each
    inside = torch.stack([chi.any(dim=1) for chi in chis]).all(dim=0)
    w = p_x * inside / (p_x * inside).sum()
    return {
        "P_all_fail": p_x[torch.stack([s == 0 for s in systems]).all(dim=0)].sum().item(),
        "P_joint_union": (p_x * inside).sum().item(),
        "label": [((w / chi.sum(dim=1).clamp(min=1))[:, None] * chi).sum(dim=0) for chi in chis],
        "closed": (w[:, None] * (states == 0)).sum(dim=0),
    }


results, traces = {}, {}
print(f"{N_CHAIN} chains, burn-in {N_BURN} sweeps, {N_KEEP} kept sweeps\n")
for case in CASES:
    name = "+".join(case)
    refs_list = [data["refs_lower"][o] for o in case]
    ex = exact(refs_list, [data["system"][o] for o in case])

    g = torch.Generator().manual_seed(0)
    t0 = time.perf_counter()
    x, gamma, c = mcmc.init_multi(refs_list, N_CHAIN)
    xs, gammas, oks = [], [], []
    for t in range(N_BURN + N_KEEP):
        x, gamma, c, ok = mcmc.step_multi(x, gamma, c, refs_list, probs, g)
        if t >= N_BURN:
            xs.append(x.argmax(dim=2).to(torch.int8))
            gammas.append(gamma.to(torch.int16))
            oks.append(ok)
    sec = time.perf_counter() - t0
    x, gamma, ok = torch.stack(xs), torch.stack(gammas), torch.stack(oks)  # (T, C, 10), (T, C, S) x2

    closed = (x == 0).double().mean(dim=(0, 1))
    rho_edge = torch.stack([mcmc.autocorrelation(x[..., i], max_lag=1)[0] for i in range(10)]).nan_to_num()
    labels = [mcmc.label_frequency(gamma[..., s].long(), len(r)).mean(dim=0) for s, r in enumerate(refs_list)]
    r = {
        "systems": list(case), "n_refs": [len(r) for r in refs_list],
        "P_all_fail": ex["P_all_fail"], "P_joint_union": ex["P_joint_union"],
        "acceptance": [mcmc.acceptance_rate(ok[..., s]).mean().item() for s in range(len(case))],
        "label_err": [(l - e).abs().max().item() for l, e in zip(labels, ex["label"])],
        "rho1_edge_max": rho_edge.max().item(),
        "closed_mcmc": closed.tolist(), "closed_exact": ex["closed"].tolist(),
        "closed_err": (closed - ex["closed"]).abs().max().item(),
        "ms_per_sweep": 1e3 * sec / (N_BURN + N_KEEP),
    }
    results[name] = r
    traces[name] = {
        "x": x, "gamma": [gamma[..., s] for s in range(len(case))],
        "boxes": [[[e for e, free in zip(edges, refs[k, :, 1]) if not free] for k in range(len(refs))]
                  for refs in refs_list],
        "label_exact": [l.tolist() for l in ex["label"]], "systems": list(case),
    }
    print(f"{name}: boxes {r['n_refs']}, P(all fail) {r['P_all_fail']:.4f}, "
          f"P(joint union) {r['P_joint_union']:.4f}, {r['ms_per_sweep']:.2f} ms/sweep")
    print(f"  acceptance per system {[round(a, 3) for a in r['acceptance']]}, "
          f"|label - pi| per system {[round(e, 4) for e in r['label_err']]}")
    print(f"  max edge lag-1 autocorrelation {r['rho1_edge_max']:.3f}, "
          f"max |P(closed | obs) error| {r['closed_err']:.4f}")
    print("  P(e_i closed | obs) exact:", dict(zip(edges, ex["closed"].numpy().round(3))))

(HERE / "results_multi.json").write_text(json.dumps(results, indent=2) + "\n")
torch.save({"edges": edges, "traces": traces, "results": results}, HERE / "traces_multi.pt")
print(f"\nwrote {HERE / 'results_multi.json'} and traces_multi.pt")
