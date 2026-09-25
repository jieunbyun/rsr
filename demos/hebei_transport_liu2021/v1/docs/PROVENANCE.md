# Provenance

This dataset is the Hebei transportation multistate network used as the third
(largest) benchmark in Liu et al. (2021).

## Relevant publications
- Liu, T., Bai, G., Tao, J., Zhang, Y.-A., & Fang, Y. (2021). An improved bounding
  algorithm for approximating multistate network reliability based on state-space
  decomposition method. *Reliability Engineering & System Safety*, 210, 107500.
  https://doi.org/10.1016/j.ress.2021.107500
- The network is originally attributed by that paper to its reference [37]:
  Bai, G., Xu, B., Chen, X., Zhang, Y., & Tao, J. (2020). *IEEE Transactions on
  Reliability*. https://doi.org/10.1109/TR.2020.3004971

## Processing notes
- The paper distributes no machine-readable data. Nodes and arcs were digitised
  from **Fig. 7** of Liu et al. (2021) (p. 8), which draws the 22 numbered nodes
  and the 33 labelled arcs `a1`–`a33` with their directions.
- Node coordinates are the centres of the drawn node circles, taken from a
  high-resolution render of the figure and normalised to unitless figure
  coordinates (figure height = 100, aspect preserved, `y` increasing northwards).
  They are schematic positions on the provincial map, **not** geodetic coordinates.
- Arc directions follow the arrowheads in the figure. Arc IDs `e01`–`e33`
  correspond one-to-one to the paper's `a1`–`a33` and are also stored in the
  `label` field of each edge.
- State distributions are taken verbatim from the paper: Section 4.1 uses
  `p = (0.05, 0.70, 0.25)` for all i.i.d. components (`probs.json`), and Table 4
  lists two further cases used in the stability study, `(0.05, 0.25, 0.70)`
  (`probs_case1.json`) and `(0.70, 0.05, 0.25)` (`probs_case3.json`).

## Verification
- The digitised arc list gives **3596** d-MPs for `d = 3` and **5665** for `d = 4`,
  matching the counts stated in Section 4.1 of the paper exactly.
- The graph is a DAG, every node lies on an s–t path, and the maximum flow with
  all arcs at capacity 2 is 6 — consistent with the demands `d = 3` and `d = 4`
  studied in the paper.
