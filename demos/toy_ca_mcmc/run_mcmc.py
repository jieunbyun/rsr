"""rsr-mcmc on bn-on-demand's toy-ca, observing SystemState::<origin> = Fail, for n3..n8.

Chains run over each origin's failure reference states (refs_lower) with the exact edge
probabilities from export_toy_ca.py. Every 1024 edge configurations are enumerated for the
exact answers, so the monitors and the posterior can be checked origin by origin:
    P(F)                 = sum of p(x) over the true failures (rules, then ndtools)
    P(union)             = sum of p(x) over the failure boxes: what the chains can reach
    P(e_i closed | F)    from the chains against the exact posterior on the union

    PYTHONPATH=. python demos/toy_ca_mcmc/run_mcmc.py      (then plot_traces.py)
"""
import json
from pathlib import Path

import torch

from rsr import mcmc

N_CHAIN, N_BURN, N_THIN, N_KEEP = 2, 200, 1, 10_000   # burn-in: several 1/(1 - rho1) steps
HERE = Path(__file__).resolve().parent
data = torch.load(HERE / "toy_ca.pt")
probs = data["probs"]                                                    # (10, 2)
states = (torch.arange(1024)[:, None] >> torch.arange(9, -1, -1)) & 1   # (1024, 10), as exported
p_x = probs[torch.arange(10), states].prod(dim=1)                        # (1024,) prior


def exact(refs, system):
    """Exact quantities by enumeration; see the module docstring."""
    chi = mcmc._membership(torch.nn.functional.one_hot(states, 2).int(), refs)   # (1024, R)
    c = chi.sum(dim=1)
    in_union = c > 0
    p_union = p_x[in_union].sum()
    w = p_x * in_union / p_union                                         # pi(x) on the union
    return {
        "P_F": p_x[system == 0].sum().item(),
        "P_union": p_union.item(),
        "union_is_failure": bool((system[in_union] == 0).all()),
        "label": ((w / c.clamp(min=1))[:, None] * chi).sum(dim=0),       # pi(gamma = r)
        "closed": (w[:, None] * (states == 0)).sum(dim=0),               # P(e_i closed | union)
    }


results, traces = {}, {}
print(f"{N_CHAIN} chains, burn-in {N_BURN}, slice {N_THIN}, {N_KEEP} kept samples per chain\n")
print("origin  R   P(F)     P(union)  P_hat(F)  accept  |label-pi|  rho1(c)  rho1(edge,max)"
      "  |P(closed|F)| max err   ms/step: proposal  accept")
for origin in data["origins"]:
    refs, system = data["refs_lower"][origin], data["system"][origin]
    ex = exact(refs, system)
    g = torch.Generator().manual_seed(0)
    n_step = N_BURN + N_THIN * N_KEEP
    x, c, gamma, ok, sec = mcmc.run(refs, probs, N_CHAIN, n_step, n_burn=N_BURN,
                                    n_thin=N_THIN, upper=False, generator=g)

    # P(union) = sum_r P(L^r) / E_pi[c]: the target is recovered from the overlap counts alone
    p_box = (probs[None] * refs).sum(dim=2).prod(dim=1)                  # (R,) P(L^r)
    p_hat = (p_box.sum() / c.double().mean()).item()
    label = mcmc.label_frequency(gamma, len(refs)).mean(dim=0)
    closed = (x[..., 0] == 1).double().mean(dim=(0, 1))                  # (10,) P(e_i closed | F)
    rho_edge = torch.stack([mcmc.autocorrelation(x[..., i, 0], max_lag=1)[0] for i in range(10)])
    rho_edge = rho_edge.nan_to_num()                                     # edges fixed in every box
    r = {
        "n_refs": len(refs), **{k: ex[k] for k in ("P_F", "P_union", "union_is_failure")},
        "P_hat": p_hat, "acceptance": mcmc.acceptance_rate(ok).mean().item(),
        "label_err": (label - ex["label"]).abs().max().item(),
        "rho1_c": mcmc.autocorrelation(c, max_lag=1)[0].nan_to_num().item(),
        "rho1_edge_max": rho_edge.max().item(),
        "closed_mcmc": closed.tolist(), "closed_exact": ex["closed"].tolist(),
        "closed_err": (closed - ex["closed"]).abs().max().item(),
        "sec": sec, "ms_per_step": {k: 1e3 * sec[k] / n_step for k in ("proposal", "accept")},
    }
    results[origin] = r
    traces[origin] = {
        "x": x.argmax(dim=3).to(torch.int8),                             # (T, C, 10), 1 = open
        "gamma": gamma.to(torch.int16),                                  # (T, C), box proposed from
        "boxes": [[e for e, free in zip(data["edges"], refs[r, :, 1]) if not free]
                  for r in range(len(refs))],                            # edges each box closes
        "label_exact": ex["label"].tolist(),                             # pi(gamma = r)
    }
    print(f"{origin:6s}  {r['n_refs']:<2d}  {r['P_F']:.4f}   {r['P_union']:.4f}    {r['P_hat']:.4f}    "
          f"{r['acceptance']:.3f}   {r['label_err']:.4f}      {r['rho1_c']:+.3f}   "
          f"{r['rho1_edge_max']:+.3f}          {r['closed_err']:.4f}                "
          f"{r['ms_per_step']['proposal']:6.2f}  {r['ms_per_step']['accept']:6.2f}")

(HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
torch.save({"edges": data["edges"], "traces": traces, "results": results}, HERE / "traces.pt")
print(f"\nwrote {HERE / 'results.json'} and traces.pt; plot with plot_traces.py")
