"""Export bn-on-demand's EMA highway example for rsr-mcmc (run in the quakenet env).

The model is built in memory from the assembly (ema-highway/gmm) with its stored rsr rules.
Writes ema.pt beside this file:
    edges      : the 129 edge ids, in the systems' component order
    probs      : (129, 2) P(EdgeOpen = False, True), exact: one bridge per edge, closed from the
                 first damage state the closure policy does not keep open; ln SA ~ N(mu, sigma^2)
                 with independent residuals and a lognormal fragility give
                 P(closed) = Phi((mu - ln theta) / sqrt(sigma^2 + beta^2))
    origins    : the 72 system origins
    refs_upper : {origin: (R, 129, 2)} survival reference states
    refs_lower : {origin: (R, 129, 2)} failure reference states

    python demos/ema_mcmc/export_ema.py --pilot ../bn-on-demand-pilot
"""
import argparse
import os
import sys
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--pilot", default=str(Path(__file__).resolve().parents[3] / "bn-on-demand-pilot"))
parser.add_argument("--check", type=int, default=200_000, help="forward samples to check P(closed)")
args = parser.parse_args()
os.chdir(args.pilot)
sys.path.insert(0, args.pilot)
from src.builder.bn_builder import assembly_path, build_tbn, compile_path  # noqa: E402

_, probs = build_tbn(compile_path(assembly_path("ema-highway/gmm")))
gmm = probs["hazard.gmm"]
site = {v.name: i for i, v in enumerate(gmm.childs)}                       # SA variable -> GMM row
systems = {k.split("::")[1]: o for k, o in probs.items() if k.startswith("SystemState::")}
edges = next(iter(systems.values())).edge_ids
assert all(o.edge_ids == edges for o in systems.values())

p_closed = []
for e in edges:
    edge = probs[f"EdgeOpen::{e}"]
    (bridge,) = edge.parents                                               # one bridge per edge
    frag = probs[bridge.name]
    i = site[frag.parents[0].name]
    k = frag.states.index(next(s for s in frag.states if s not in edge.usable_states))
    theta, beta = frag.medians[k - 1], frag.betas[k - 1]                   # limit state DS >= k
    z = (gmm.ln_median[i] - torch.log(theta)) / torch.sqrt(gmm.sigma[i] ** 2 + beta ** 2)
    p_closed.append(torch.special.ndtr(z.double()))
p_closed = torch.stack(p_closed)

out = {"edges": edges, "probs": torch.stack([p_closed, 1 - p_closed], dim=1), "origins": list(systems),
       "refs_upper": {o: s.rules[0].int().cpu() for o, s in systems.items()},
       "refs_lower": {o: s.rules[1].int().cpu() for o, s in systems.items()}}
path = Path(__file__).resolve().with_name("ema.pt")
torch.save(out, path)

if args.check:
    from tbnpy import inference
    torch.manual_seed(0)
    keys = [k for k in probs if not k.startswith("SystemState::")]
    S = inference.sample(probs, keys, n_sample=args.check)
    emp = torch.stack([(S[f"EdgeOpen::{e}"].Cs[:, 0] == 0).double().mean() for e in edges])
    se = (p_closed * (1 - p_closed) / args.check).sqrt()
    print(f"P(closed) exact vs {args.check} forward samples: max |diff| {(emp - p_closed).abs().max():.5f}, "
          f"max |diff| / SE {((emp - p_closed).abs() / se.clamp(min=1e-12)).max():.2f}")
print(f"{len(edges)} edges, P(closed) from {p_closed.min():.4f} to {p_closed.max():.4f}; "
      f"{len(systems)} origins -> {path}")
