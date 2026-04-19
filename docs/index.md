# RSR — Rule-based System Reliability

## What it's for

- Fast system probability calculation of coherent systems.
- A representative class is **network systems**.
- Yet there are other systems, including k-out-of-N systems, structural systems, series/parallel systems, and series-parallel systems.
- RSR can use **CUDA** — if a computing GPU is available, the computation can become faster.

## Workflow

Prepare inputs (system function and probabilities) → obtain reference states → compute system probabilities.

## Demonstrations

See the [`demos/`](https://github.com/jieunbyun/rsr/tree/main/demos) folder:

- A toy network's global connectivity.
- Eastern Massachusetts (EMA) highway benchmark network's shortest path.
- EMA highway benchmark network's accessible population.
- Random graphs' connectivity and global connectivity.

```{toctree}
:maxdepth: 2
:caption: Contents

usage
api
contact
```
