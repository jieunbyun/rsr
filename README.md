# RSR: Reference-state System Reliability Method

## Overview
This repository provides a Python implementation of **RSR (Reference-state System Reliability Method)** for
efficient reliability and resilience analysis of networks. It includes:
- Core package `rsr/` with tensor-based algorithms
- Demonstration notebooks under `demos/`
- Unit tests under `tests/`

The code is designed for research and education on large-scale system uncertainty quantification.

## Publication / Citation
In preparation. Most relevant publication is: Byun, J. E., Ryu, H. & Straub, D. (2024). Branch-and-bound algorithm for efficient reliability analysis of general coherent systems. arXiv preprint arXiv:2410.22363.

## Features
- Reference-state system reliability and rule extraction algorithms
- Example benchmark datasets on various systems (e.g., distribution substation, EMA shortest path, toy k-connectivity)
- The network data in the demos are from GitHub repo [network-datasets](https://github.com/jieunbyun/network-datasets)
- PyTorch-friendly implementations for scalable computation


## Installation

You can install `rsr` in one of two ways.

### Option 1: Install from PyPI

The package is published on PyPI under the distribution name `rsr-duco` (the name `rsr` was already taken). The Python import name is still `rsr`:

```bash
pip install rsr-duco
```

### Option 2: Install from source (developer version)

Clone the repo and install in editable mode (useful for development or when you want to modify the code):

```bash
git clone https://github.com/jieunbyun/rsr.git
cd <path/to/rsr>
pip install -e .
```

### Using the package

Either option gives you the same import name:

```python
import rsr
from rsr import rsr, utils
```

Dependencies are listed in `pyproject.toml`.

## Usage
Refer to the demonstration notebooks in `demos/` for example workflows:

## License
This project is licensed under the terms of the LICENSE file included in this repository.
