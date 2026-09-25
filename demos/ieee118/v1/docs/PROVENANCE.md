# Provenance

## Network topology and electrical parameters

`data/ieee118.m` is the standard IEEE 118-bus test case, distributed with
MATPOWER. The bus, branch, and generator records (admittances, limits, demand,
generator cost) come from this file unmodified.

## Component failure model

The 4-state generator / 2-state ordinary-bus / 2-state branch failure model and
the 13.8% blackout threshold (Scenario 1) follow:

> Chan, J., Papaioannou, I., & Straub, D. (2024).
> An adaptive subset simulation algorithm for system reliability analysis with
> discontinuous limit states.
> *Reliability Engineering & System Safety*, 110009.

The same component model is reused in:

> Byun, J.-E., Ryu, H., & Straub, D. (2024).
> Branch-and-bound algorithm for efficient reliability analysis of general
> coherent systems. arXiv:2410.22363.

which reports a reference failure probability of ≈ 1.0 × 10⁻⁴ for this model.

## Data files in this dataset

The JSON files (`nodes.json`, `edges.json`, `probs.json`) and the DC-OPF
system-function script (`sfun_dcopt.py`, `func_dcopt_py.py`, `run_case118.py`)
were prepared by the authors of Byun et al. (2024) and mirrored here from the
TSUM demo repository (`tsum/demos/case118`).

## License

CC-BY-4.0. Please cite Byun et al. (2024) and Chan et al. (2024) when using
this dataset.
