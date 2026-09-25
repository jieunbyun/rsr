# Dataset: hebei-transport-liu2021 (v1.0.0)

## Summary
**Hebei transportation multistate network**, the largest benchmark used by
Liu et al. (2021) to test two-terminal multistate network reliability algorithms.
It has **22 nodes** and **33 directed arcs**, every arc being an i.i.d. component
with **three capacity states (0, 1, 2)**. Reliability is defined as the probability
that the maximum flow from the source to the sink is at least the demand `d`;
the reference studies `d = 3` and `d = 4`.

- Source: `n1` (node **1**, CD, drawn as **s** in the paper)
- Sink: `n22` (node **22**, HD, drawn as **t**)
- Maximum flow with all arcs at capacity 2: **6**

## Structure
- `data/nodes.json`
  ```json
  "n1": { "x": 50.069, "y": 73.651, "label": "CD", "city": "Chengde", "role": "source" }
  ```

- `data/edges.json` — directed arcs (`directed: true`), `label` gives the arc name
  used in Fig. 7 of the paper (`e01` ↔ `a1`, …, `e33` ↔ `a33`).
  ```json
  "e01": { "from": "n1", "to": "n2", "directed": true, "label": "a1",
           "n_states": 3, "capacities": [0, 1, 2] }
  ```

- `data/probs.json` — the state distribution used in Section 4.1 of the paper,
  `p = (0.05, 0.70, 0.25)` for every arc (this is "Case 2" of Table 4).
  ```json
  "e01": { "0": {"p": 0.05, "capacity": 0},
           "1": {"p": 0.70, "capacity": 1},
           "2": {"p": 0.25, "capacity": 2} }
  ```

- `data/probs_case1.json` — Case 1 of Table 4, `p = (0.05, 0.25, 0.70)`.
- `data/probs_case3.json` — Case 3 of Table 4, `p = (0.70, 0.05, 0.25)`.

## Data Dictionary
### Nodes (`nodes.json`)
- `x`, `y` — schematic coordinates digitised from Fig. 7 of the paper. They are
  **unitless figure coordinates** (not geodetic), normalised so that the figure
  height spans 0–100 with the aspect ratio preserved; `y` increases northwards.
- `label` — city abbreviation printed in the figure, or `null` for the eight
  unlabelled junction nodes (6, 9, 13, 16, 17, 19, 20, 21).
- `city` — expanded city name (`null` where `label` is `null`); see the note below.
- `role` — `"source"` on `n1`, `"sink"` on `n22`, absent otherwise.

### Edges (`edges.json`)
- `from`, `to` — node IDs; arcs are **directed** (`directed: true`).
- `label` — arc name in Fig. 7 of the paper (`a1`–`a33`).
- `n_states` — 3 for every arc.
- `capacities` — the capacity carried in each state, i.e. state `k` carries `k` units.

### Probabilities (`probs*.json`)
- Keyed by edge ID, then by state index `"0"`, `"1"`, `"2"`, with `p` and `capacity`.
- All 33 arcs are i.i.d., so every arc gets the same distribution.

## Usage
```python
from pathlib import Path
import json

root = Path("datasets/hebei_transport_liu2021/v1/data")

nodes = json.loads((root / "nodes.json").read_text("utf-8"))
edges = json.loads((root / "edges.json").read_text("utf-8"))
probs = json.loads((root / "probs.json").read_text("utf-8"))
```

## Notes
- City abbreviations are as printed in Fig. 7. The expansions in `city` are the
  standard Hebei/Beijing–Tianjin city names (CD Chengde, ZJK Zhangjiakou,
  BJ Beijing, QHD Qinhuangdao, TS Tangshan, TJ Tianjin, BZ Bazhou, BD Baoding,
  HH Huanghua, CZ Cangzhou, HS Hengshui, SJZ Shijiazhuang, XT Xingtai,
  HD Handan). The paper does not spell them out, so `BZ` and `HH` in particular
  are inferred.
- The reference reports **3596** d-MPs for `d = 3` and **5665** for `d = 4`; both
  counts reproduce exactly on this arc list, which confirms the digitisation.
