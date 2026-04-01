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
Clone the repo and install dependencies:
```bash
git clone https://github.com/jieunbyun/rsr.git
cd <path/to/rsr>
pip install -e .
```

Then you can import the package in Python:
```python
import rsr
from rsr import rsr, utils
```

Dependencies are listed in `pyproject.toml`.

## Usage
Refer to the demonstration notebooks in `demos/` for example workflows:

## License
This project is licensed under the terms of the LICENSE file included in this repository.
