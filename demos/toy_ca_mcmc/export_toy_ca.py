"""Export bn-on-demand's toy-ca example for rsr-mcmc (run in the quakenet env).

Writes toy_ca.pt beside this file:
    edges      : edge ids e1..e10, the component order
    probs      : (10, 2) P(EdgeOpen = False, True), exact: one bridge per edge, closed at
                 extensive damage, ln SA ~ N(mu, sigma^2) and P(DS >= ext | SA) lognormal, so
                 P(closed) = Phi((mu - ln theta_ext) / sqrt(sigma^2 + beta_ext^2))
    origins    : n3..n8
    refs_upper : {origin: (R, 10, 2)} survival reference states, as stored by bn-on-demand
    refs_lower : {origin: (R, 10, 2)} failure reference states
    system     : {origin: (1024,)} true system state of every edge configuration (0 Fail,
                 1 Survival), from rules then ndtools; row k is the binary code of k, e1 first

    python demos/toy_ca_mcmc/export_toy_ca.py --pilot ../bn-on-demand-pilot
"""
import argparse
import sys
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--pilot", default=str(Path(__file__).resolve().parents[3] / "bn-on-demand-pilot"))
args = parser.parse_args()
sys.path.insert(0, str(Path(args.pilot) / "examples/toy-ca/tbnpy/demo"))
from model import define_model  # noqa: E402

_, probs = define_model()
gmm = probs["hazard.gmm"]
bridges = [v.name.split("::")[1] for v in gmm.childs]                   # b1..b10, edge ei carries bi
edges = ["e" + b[1:] for b in bridges]

p_closed = []
for i, b in enumerate(bridges):
    frag = probs[f"DamageState.HazusBridge::{b}"]
    usable = probs[f"EdgeOpen::{edges[i]}"].usable_states
    k = frag.states.index(next(s for s in frag.states if s not in usable))   # first closing state
    theta, beta = frag.medians[k - 1], frag.betas[k - 1]                 # limit state DS >= k
    z = (gmm.ln_median[i] - torch.log(theta)) / torch.sqrt(gmm.sigma[i] ** 2 + beta ** 2)
    p_closed.append(torch.special.ndtr(z.double()))
p_closed = torch.stack(p_closed)

all_states = (torch.arange(1024)[:, None] >> torch.arange(9, -1, -1)) & 1   # (1024, 10), e1 = MSB
origins = [k.split("::")[1] for k in probs if k.startswith("SystemState::")]
out = {"edges": edges, "probs": torch.stack([p_closed, 1 - p_closed], dim=1), "origins": origins,
       "refs_upper": {}, "refs_lower": {}, "system": {}}
for o in origins:
    sys_prob = probs[f"SystemState::{o}"]
    assert sys_prob.edge_ids == edges
    out["refs_upper"][o], out["refs_lower"][o] = (r.int().cpu() for r in sys_prob.rules)
    out["system"][o] = sys_prob.evaluate(all_states.double())[:, 0].long().cpu()

path = Path(__file__).with_name("toy_ca.pt")
torch.save(out, path)
print("P(closed):", dict(zip(edges, p_closed.numpy().round(4))))
print({o: (len(out["refs_upper"][o]), len(out["refs_lower"][o])) for o in origins}, "->", path)
