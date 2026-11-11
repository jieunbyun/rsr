# EMA highways: MCMC given two observed system failures

This demo samples the edge states of a highway network given that two of its systems were
observed to fail. It uses `rsr.mcmc` and checks the result against an independent reference.

- **Network**: bn-on-demand's EMA highway example, with 129 edges, each either open or closed.
- **Observation**: the systems for origins `n57` and `n58` both failed.
- **Why this case**: each system has thousands of failure boxes (2,451 and 2,665), and the two
  sets overlap only in a few small regions. A chain that moves one system's box at a time
  rarely crosses between those regions. The tuple jump, after each sweep, moves both boxes
  at once.
- **Check**: the chains are compared against a weighted importance-sampling reference, which
  draws 4 million states directly from where the two systems' boxes intersect.

It takes about 1.5 minutes with a CUDA GPU, or about 6 minutes on a CPU. Read
[../toy_ca_mcmc/README.md](../toy_ca_mcmc/README.md) first for a 20-second version on a
10-edge network.

## 1. Set up the environment

You need [conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda is enough) and git.
Run these in a terminal (Anaconda Prompt on Windows):

```bash
conda create -n rsr python=3.12 -y
conda activate rsr

git clone https://github.com/jieunbyun/rsr.git
cd rsr
pip install -e .
```

`pip install -e .` installs `rsr` from this folder together with its dependencies (PyTorch,
NumPy, matplotlib and others), so the demo runs the code you cloned.

**Optional, to use an NVIDIA GPU.** The PyTorch that pip installs on Windows runs on the CPU
only. For the GPU version, take the install command for your CUDA version from
[pytorch.org](https://pytorch.org/get-started/locally/), for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

To check the install, and whether a GPU will be used:

```bash
python -c "import torch; from rsr import mcmc; print('ok; GPU:', torch.cuda.is_available())"
```

The demo uses the GPU when `torch.cuda.is_available()` is `True`, and the CPU otherwise.

## 2. Run the demo

From the repository root (the `rsr` folder you cloned), with the environment active:

```bash
python demos/ema_mcmc/run.py
python demos/ema_mcmc/plot.py
```

`run.py` first builds the reference, then runs the chains, and prints a summary like this
(on a GPU):

```
EMA, n57 failed, n58 failed; boxes [2451, 2665]; on cuda
reference: 4,000,000 weighted-IS draws from the box-pair intersections, P(E) = 1.474e-07, ESS 3,959,150, 7 s
1000 chains, 300 + 1000 sweeps, each followed by a tuple jump (m = 8, gamma = 1, random order), sequential start; 83 s
P(e closed | E): max error 0.0256 against the reference (0/129 edges off by > 0.05)
R-hat max 1.166, ESS min 21808; P(E) from the sequential start / reference 0.951
...
```

In a correct run, the largest error in P(edge closed) is about 0.03 or less and no edge is
off by more than 0.05. R-hat around 1.2 is expected here: the chains cross between the
overlap regions only occasionally, by tuple jumps, in about 0.3% of sweeps. The CPU and the GPU
draw different random numbers, so the figures differ between them: one CPU run gave a largest
error of 0.015 and R-hat 1.17, in 39 s for the reference and 300 s for the chains. The run
time also depends on your machine.

## 3. Outputs

`run.py` and `plot.py` write these files into this folder:

| File | Contents |
|---|---|
| `run.log` | The printed summary, including the time each step of a sweep takes. |
| `results.json` | The summary and each edge's posterior, from the chains and from the reference. |
| `reference.pt` | The reference, cached: a second run reuses it. Not tracked by git. |
| `traces.pt` | The per-sweep traces that `plot.py` reads. Not tracked by git. |
| `posterior.png` | Each edge's P(closed), MCMC against the reference, with bars of ±2 reference standard errors. The points should lie on the diagonal. |
| `traces.png` | Two chains after burn-in: the box each holds for `n57` and `n58` (triangles mark tuple jumps to a partner), and the 25 edges with the most uncertain posterior. |
| `traces_all.png` | All 1,000 chains' boxes, burn-in included, and the share of chains with each of those edges closed at each sweep beside the reference. |

Running the scripts replaces the copies of these files that come with the repository.

## Trying other settings

The settings are at the top of `run.py`:

```python
OBSERVED = {"n57": 0, "n58": 0}          # 0 = failed, 1 = survived
N_CHAIN, N_BURN, N_KEEP, N_PARTNER, GAMMA, N_TRACE = 1000, 300, 1000, 8, 1.0, 2
N_REF, BATCH, N_GROUP, ROW_CHUNK = 4_000_000, 100_000, 20, 32
```

- **Faster check**: for a quicker run, lower `N_CHAIN` (for example to 200) and `N_KEEP`.
- **Tuple jump**: `N_PARTNER` is the number of partner tuples per jump.
- **Observation**: only `n57` and `n58` are in the data file. `0` means failed and `1` means
  survived.
- **Reference limitation**: the reference draws from the intersection of the two observed
  systems' boxes, so it only applies to a pair of observed systems.

## Data

`ema_n57_n58.pt` holds the 129 edges' closure probabilities and the reference states
(survival and failure boxes) of origins n57 and n58. It was made with

```bash
python demos/ema_mcmc/export_ema.py --pilot ../bn-on-demand-pilot --origins n57 n58
```

from the bn-on-demand pilot repository, using its `quakenet` environment. You don't need that
repository to run the demo. Without `--origins`, the script exports every origin to `ema.pt`
(about 100 MB, not tracked by git).
