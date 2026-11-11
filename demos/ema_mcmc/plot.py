"""Figures of run.py: posterior.png, traces.png and traces_all.png, drawn by the functions of
../toy_ca_mcmc/plot.py (see there).

    python demos/ema_mcmc/plot.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "toy_ca_mcmc"))
from plot import main  # noqa: E402

main(HERE)
