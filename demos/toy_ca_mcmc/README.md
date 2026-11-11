# toy-ca: MCMC given two observed system failures

This demo samples the edge states of a small road network given that two of its systems were
observed to fail. It uses `rsr.mcmc` and checks the result against the exact answer.

- **Network**: bn-on-demand's toy-ca example, with 10 edges, each either open or closed.
- **Observation**: the systems for origins `n4` and `n6` both failed.
- **Method**: each sweep updates every observed system's reference box and the edge states,
  then makes a tuple jump that can move both systems' boxes at once.
- **Check**: with only 10 edges, all 1,024 edge states can be enumerated. That gives the
  exact posterior, which the chains are compared against.

It takes about 20 seconds on a laptop CPU, and needs no GPU.

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
NumPy, matplotlib and others), so the demo runs the code you cloned. To check the install:

```bash
python -c "from rsr import mcmc; print('ok')"
```

## 2. Run the demo

From the repository root (the `rsr` folder you cloned), with the environment active:

```bash
python demos/toy_ca_mcmc/run.py
python demos/toy_ca_mcmc/plot.py
```

`run.py` runs the chains and prints a summary like this:

```
toy-ca, n4 failed, n6 failed; boxes [4, 2]; P(E) = 0.03015 (exact)
1000 chains, 200 + 2000 sweeps, each followed by a tuple jump (m = 8, gamma = 1, random order), sequential start; 18.2 s on cpu
P(e closed | E): max error 0.0006 against the exact posterior; TV over the 1024 states 0.0013
R-hat max 1.0002, ESS min 1162490; P(E) from the sequential start / exact 1.0035
...
```

In a correct run, the largest error in P(edge closed) is below about 0.002 and R-hat is close
to 1. The run time and the last digits depend on your machine.

## 3. Outputs

`run.py` and `plot.py` write these files into this folder:

| File | Contents |
|---|---|
| `run.log` | The printed summary, including the time each step of a sweep takes. |
| `results.json` | The summary and each edge's posterior, from the chains and exact. |
| `traces.pt` | The per-sweep traces that `plot.py` reads. Not tracked by git. |
| `posterior.png` | Each edge's P(closed), MCMC against exact. The points should lie on the diagonal. |
| `traces.png` | Two chains after burn-in: the box each holds for `n4` and `n6`, and every edge's state. |
| `traces_all.png` | All 1,000 chains' boxes, burn-in included, and the share of chains with each edge closed at each sweep beside the exact value. |

Running the scripts replaces the copies of these files that come with the repository.

## Trying another case

The settings are at the top of `run.py`:

```python
OBSERVED = {"n4": 0, "n6": 0}            # 0 = failed, 1 = survived
N_CHAIN, N_BURN, N_KEEP, N_PARTNER, GAMMA, N_TRACE = 1000, 200, 2000, 8, 1.0, 2
```

- **Observed systems**: any of the origins `n3` to `n8`, each failed (0) or survived (1).
  For example, `{"n4": 0, "n7": 1}` means n4 failed and n7 survived.
- **Run length**: `N_CHAIN` is the number of chains, and `N_BURN` and `N_KEEP` the sweeps
  discarded and kept.
- **Tuple jump**: `N_PARTNER` is the number of partner tuples per jump.

## Data

`toy_ca.pt` holds the edges' closure probabilities and every origin's reference states
(survival and failure boxes). `export_toy_ca.py` made it from the bn-on-demand pilot
repository. You don't need that repository to run the demo.
