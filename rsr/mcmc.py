"""Certificate-augmented MCMC over reference boxes, for independent multi-state components
("Box jumps under component independence", 9 November 2026).

Systems j = 1..J are observed, each through the reference boxes of its observed state: lower
(failure) boxes for a failed system, upper (survival) boxes for a surviving one. A tuple
t = (lambda_1, ..., lambda_J) picks one box per system, and its composed box is their
intersection (Eq. 5). The chains target p(x) restricted to E, the intersection over systems of
the union of each system's boxes (Eq. 3), with t uniform over the tuples whose box contains x
(Eq. 8).

Shapes, for C chains, n components and M states (probs padded with zeros where a component has
fewer states):
    probs : (n, M)          independent prior, probs[i, s] = P(X_i = s)
    x     : (C, n) long     component states
    t     : (C, J) long     labels, one box per system
    chi   : list of (C, R_j) bool, chi[j][c, lam] = 1(x_c in box lam of system j)   (Eq. 6)

    mdl = model(refs_list, probs)      # refs: rsr masks (R_j, n, M), or (lo, hi) corners (R_j, n)
    trace = run(mdl, n_chain=1000, n_sweep=1300, n_burn=300)
    acceptance_rate(trace), label_change_rate(trace), occupancy(trace, mdl), mixing(trace, M),
    mean_counts(trace), timing(trace)

The numbered functions follow the scheme of Section 7:
    1  initialise       sequential draw of a feasible start, and an estimate of P(E) (Section 11)
    2  system_order     random order of the systems, drawn without looking at the state
    3a leave_one_out    box held by the other systems
    3b draw_label       new label of system j, by its conditional box probability (Eq. 11, 13)
    4  draw_state       x' ~ p(. | composed box)                                   (Eq. 4)
    5  accept           min(1, c(x) / c(x')), else keep (x, t)                     (Eq. 12)
    6  refresh_labels   every label uniform over the boxes containing x            (Eq. 9)
    7  run              sweeps of 2-6, recording x and t whether accepted or not
    8-12                reports: acceptance and label-change rates per system, occupancy of every
                        box, autocorrelation / ESS / R-hat of every 1(X_i = s), mean counts
    13 timing           computation time per step of the sweep, and per sweep

The tuple jump (Section 4, "Tuple jump") changes every label in one step, with partner tuples
from the distribution rho of Eq. (rho) (partner_rho: an order of the systems, by default a random
permutation per jump, the first label uniform, each later one by its conditional box
probability^gamma):
    T1 draw_partners    tau = (t, t^1, ..., t^m), the held tuple and m partners drawn from rho
    T2 choose_tuple     k with probability P(L^{tau^k}) / sum_l P(L^{tau^l})
    T3 draw_state       x' ~ p(. | L^{tau^k}), as step 4
    T4 accept_tuple     min{1, c(x)/c(x') * c_tau(x)/c_tau(x') * s_tau(x')/s_tau(x)}
       tuple_jump       T1-T4, then the labels refreshed (Eq. 9); run(..., n_partner=m) follows
                        every sweep with one
"""

import math
import time
from typing import NamedTuple

import torch

_OUT_CHUNK = 1 << 26                   # elements of a (chains, boxes) product held at once


class Boxes(NamedTuple):
    """One system's reference boxes [lo, hi], (R, n) each. Their constrained coordinates are
    gathered into Q columns, one per distinct (component, interval) that some box imposes:
    column q holds component col_comp[q] in [col_lo[q], col_hi[q]], and inc[r, q] = 1 if box r
    imposes column q. A coordinate is constrained if its interval is not all of {0, ..., M - 1};
    a box imposes at most one column per component. inc64_t, inc32_t are inc transposed (Q, R)."""
    lo: torch.Tensor
    hi: torch.Tensor
    col_comp: torch.Tensor
    col_lo: torch.Tensor
    col_hi: torch.Tensor
    inc64_t: torch.Tensor
    inc32_t: torch.Tensor


class Model(NamedTuple):
    """The observed systems' boxes and the prior. cdf[i, s + 1] = F_i(s) and cdf[i, 0] = 0 (Eq. 1)."""
    systems: list
    probs: torch.Tensor
    cdf: torch.Tensor


def boxes(lo, hi, n_state):
    """Boxes [lo, hi] from their corners (R, n): lower reference states l give lo = 0, hi = l,
    upper reference states u give lo = u, hi = M - 1."""
    lo, hi = lo.long(), hi.long()
    if (lo > hi).any() or (lo < 0).any() or (hi > n_state - 1).any():
        raise ValueError("box corners must satisfy 0 <= lo <= hi <= M - 1")
    pb, pc = torch.nonzero((lo > 0) | (hi < n_state - 1), as_tuple=True)
    key = (pc * n_state + lo[pb, pc]) * n_state + hi[pb, pc]
    cols, col_of_pair = torch.unique(key, return_inverse=True)
    inc = torch.zeros(len(lo), len(cols), dtype=torch.float64, device=lo.device)
    inc[pb, col_of_pair] = 1.0
    return Boxes(lo, hi, cols // n_state ** 2, cols // n_state % n_state, cols % n_state,
                 inc.T.contiguous(), inc.T.float().contiguous())


def boxes_from_masks(refs):
    """Boxes from rsr reference states (R, n, M), refs[r, i, s] = 1 if box r allows X_i = s."""
    allowed = refs.bool()
    M = allowed.shape[2]
    lo = allowed.int().argmax(dim=2)
    hi = M - 1 - allowed.flip(2).int().argmax(dim=2)
    if not torch.equal(allowed.sum(dim=2), hi - lo + 1):
        raise ValueError("a reference state allows a set of states that is not an interval")
    return boxes(lo, hi, M)


def model(refs_list, probs):
    """refs_list holds, per observed system, rsr masks (R_j, n, M) or a pair (lo, hi) of corners."""
    probs = probs.double()
    dev, M = probs.device, probs.shape[1]
    systems = [boxes(r[0].to(dev), r[1].to(dev), M) if isinstance(r, (tuple, list))
               else boxes_from_masks(r.to(dev)) for r in refs_list]
    cdf = torch.cat([torch.zeros_like(probs[:, :1]), probs.cumsum(dim=1)], dim=1)
    return Model(systems, probs, cdf / cdf[:, -1:])


def _mass(cdf, comp, lo, hi):
    """P(lo <= X_comp <= hi) elementwise, for lo <= hi."""
    W = cdf.shape[1]
    flat = cdf.reshape(-1)
    return flat[comp * W + hi + 1] - flat[comp * W + lo]


def _categorical(w, n_draw, generator=None):
    """n_draw indices, row c drawn with probability w[c] / w[c].sum(), by inverse CDF; w is
    (n_draw, R), or (1, R) shared by every draw. Zero weights are never drawn."""
    cum = w.double().cumsum(dim=1)
    u = (1 - torch.rand(n_draw, 1, dtype=torch.float64, device=w.device, generator=generator)) * cum[:, -1:]
    if len(cum) == 1:
        idx = torch.searchsorted(cum[0], u[:, 0])
    else:
        idx = torch.searchsorted(cum, u).squeeze(1)
    return idx.clamp(max=w.shape[1] - 1)


def _row_chunks(n_row, n_box):
    size = max(1, _OUT_CHUNK // max(n_box, 1))
    return [slice(k, k + size) for k in range(0, n_row, size)]


def membership(mdl, x):
    """chi_j(x), (C, R_j) bool for every system j (Eq. 6): x violates none of the columns its box
    imposes. One product (C, Q_j) @ (Q_j, R_j) counts the violations; exact in float32."""
    out = []
    for b in mdl.systems:
        chi = torch.empty(len(x), len(b.lo), dtype=torch.bool, device=x.device)
        for rows in _row_chunks(len(x), len(b.lo)):
            xs = x[rows][:, b.col_comp]
            viol = ((xs < b.col_lo) | (xs > b.col_hi)).float()                 # (C, Q)
            chi[rows] = (viol @ b.inc32_t) == 0
        out.append(chi)
    return out


def counts(chi):
    """c_j(x) = number of system j's boxes containing x, (C, J); c(x) is their product (Eq. 7)."""
    return torch.stack([ch.sum(dim=1) for ch in chi], dim=1)


_GRID = 2.0 ** -8                      # float32 sums of multiples of _GRID are exact below 2^15


def _sum_columns(v, bx):
    """v @ inc^T for float64 v (C, Q): the sum over the columns each box imposes, (C, R).
    GPUs multiply float64 slowly, so there v is split into k * 2^-8, whose float32 sums are
    exact while every |row sum| < 2^15 (checked), and a remainder below 2^-9, whose float32 sum
    is off by about 1e-9. Elsewhere, or with TF32 products enabled, the product is float64."""
    if v.device.type == "cuda" and not torch.backends.cuda.matmul.allow_tf32:
        if v.abs().sum(dim=1).max().item() < 2.0 ** 15:
            grid = torch.round(v / _GRID) * _GRID
            return ((grid.float() @ bx.inc32_t).double()
                    + ((v - grid).float() @ bx.inc32_t).double())
    return v @ bx.inc64_t


def label_logprob(mdl, j, a, b):
    """log P(L^{j,lam} | [a, b]) for every box lam of system j, (C, R_j), with a, b (C, n) the
    corners of the box held by the other systems (Eq. 13); -inf where box lam misses [a, b].
    Only the coordinates box lam constrains contribute a factor other than one, so with
    H[c, q] = log P(column q's interval | [a_c, b_c]) on its component, the result is
    H @ inc^T, a (C, Q_j) @ (Q_j, R_j) product; a second product counts the empty factors."""
    bx = mdl.systems[j]
    out = torch.empty(len(a), len(bx.lo), dtype=torch.float64, device=a.device)
    for rows in _row_chunks(len(a), len(bx.lo)):
        A, B = a[rows][:, bx.col_comp], b[rows][:, bx.col_comp]                 # (C, Q)
        lo, hi = torch.maximum(A, bx.col_lo), torch.minimum(B, bx.col_hi)
        num = _mass(mdl.cdf, bx.col_comp, lo, torch.maximum(hi, lo))
        den = _mass(mdl.cdf, bx.col_comp, A, B)
        empty = (lo > hi) | (num <= 0)
        log_f = torch.where(empty, 0.0, num.clamp(min=1e-300).log() - den.log())
        lp = _sum_columns(log_f, bx)
        out[rows] = torch.where((empty.float() @ bx.inc32_t) > 0, -torch.inf, lp)
    return out


# --- 1. initialisation -------------------------------------------------------------------------

def initialise(mdl, n_chain, generator=None, max_try=100):
    """Sequential draw (Section 11): per chain, a random order of the systems; each system's label
    drawn by its probability given the boxes chosen so far (Eq. 22); x ~ p(. | L^t); labels
    refreshed (Eq. 9). A chain whose partial tuple no box of the next system meets (possible when
    failures and survivals are observed together) starts again, and counts as a draw of weight 0.
    Returns x, t, chi and p_event, the estimate sum w / draws of P(E) with w = prod zeta / c(x)
    (Lemma 2)."""
    dev, J = mdl.probs.device, len(mdl.systems)
    n, M = mdl.probs.shape
    x = torch.zeros(n_chain, n, dtype=torch.long, device=dev)
    t = torch.zeros(n_chain, J, dtype=torch.long, device=dev)
    log_w = torch.full((n_chain,), -torch.inf, dtype=torch.float64, device=dev)
    pending = torch.ones(n_chain, dtype=torch.bool, device=dev)
    n_draw = 0
    for _ in range(max_try):
        idx = torch.nonzero(pending).squeeze(1)
        C = len(idx)
        n_draw += C
        order = torch.rand(C, J, device=dev, generator=generator).argsort(dim=1)    # state-free
        a = torch.zeros(C, n, dtype=torch.long, device=dev)
        b = torch.full((C, n), M - 1, dtype=torch.long, device=dev)
        lam_all = torch.zeros(C, J, dtype=torch.long, device=dev)
        lw = torch.zeros(C, dtype=torch.float64, device=dev)
        alive = torch.ones(C, dtype=torch.bool, device=dev)
        for k in range(J):
            for j in range(J):
                sel = torch.nonzero((order[:, k] == j) & alive).squeeze(1)
                if len(sel) == 0:
                    continue
                logq = label_logprob(mdl, j, a[sel], b[sel])
                log_zeta = torch.logsumexp(logq, dim=1)
                dead = torch.isinf(log_zeta)
                alive[sel[dead]] = False
                sel, logq, log_zeta = sel[~dead], logq[~dead], log_zeta[~dead]
                if len(sel) == 0:
                    continue
                lam = _categorical((logq - log_zeta[:, None]).exp(), len(sel), generator)
                lam_all[sel, j] = lam
                lw[sel] += log_zeta
                bx = mdl.systems[j]
                a[sel] = torch.maximum(a[sel], bx.lo[lam])
                b[sel] = torch.minimum(b[sel], bx.hi[lam])
        done = torch.nonzero(alive).squeeze(1)
        if len(done):
            xs = draw_state(mdl, a[done], b[done], generator)
            c = counts(membership(mdl, xs))
            x[idx[done]], t[idx[done]] = xs, lam_all[done]
            log_w[idx[done]] = lw[done] - c.double().log().sum(dim=1)
            pending[idx[done]] = False
        if not pending.any():
            break
    else:
        raise RuntimeError(f"no feasible start for {int(pending.sum())} chains in {max_try} tries")
    chi = membership(mdl, x)
    t = refresh_labels(chi, generator)
    p_event = (torch.logsumexp(log_w, dim=0) - math.log(n_draw)).exp().item()
    return x, t, chi, p_event


# --- 2. order of the systems -------------------------------------------------------------------

def system_order(n_system, generator=None, device="cpu"):
    """A uniformly random permutation of the systems. It takes no state, so the order cannot
    depend on it (Section 10, state-independent order)."""
    return torch.randperm(n_system, generator=generator, device=device)


# --- 3a. box held by the other systems ---------------------------------------------------------

def leave_one_out(mdl, t, j):
    """Corners a, b (C, n) of L^{t_-j}, the intersection of the boxes the other systems hold;
    the whole space when J = 1."""
    n, M = mdl.probs.shape
    a = torch.zeros(len(t), n, dtype=torch.long, device=t.device)
    b = torch.full((len(t), n), M - 1, dtype=torch.long, device=t.device)
    for k, bx in enumerate(mdl.systems):
        if k != j:
            a, b = torch.maximum(a, bx.lo[t[:, k]]), torch.minimum(b, bx.hi[t[:, k]])
    return a, b


# --- 3b. new label -----------------------------------------------------------------------------

def draw_label(mdl, j, a, b, generator=None, logq=None):
    """lam' with probability P(L^{j,lam'} | [a, b]) / zeta_j (Eq. 11); returns lam' (C,) and
    log zeta_j (C,), zeta_j = sum over the boxes of system j of P(L^{j,mu} | [a, b]).
    logq (1, R_j), if given, is label_logprob shared by every chain (one observed system,
    where [a, b] is always the whole space), and is not recomputed."""
    if logq is None:
        logq = label_logprob(mdl, j, a, b)
    top = logq.max(dim=1, keepdim=True).values
    if torch.isinf(top).any():
        raise ValueError(f"no box of system {j} meets the box held by the other systems")
    w = (logq - top).exp()                                                  # one pass for both uses
    log_zeta = top[:, 0] + w.sum(dim=1).log()
    lam = _categorical(w, len(a), generator)
    return lam, log_zeta.expand(len(a))


# --- 4. new state ------------------------------------------------------------------------------

def draw_state(mdl, lo, hi, generator=None):
    """x' ~ p(. | lo <= x <= hi) by independent truncated draws, one per component (Eq. 4)."""
    C, n = lo.shape
    W = mdl.cdf.shape[1]
    comp = torch.arange(n, device=lo.device) * W
    flat = mdl.cdf.reshape(-1)
    F_lo, F_hi = flat[comp + lo], flat[comp + hi + 1]
    u = F_lo + (1 - torch.rand(C, n, dtype=torch.float64, device=lo.device, generator=generator)) * (F_hi - F_lo)
    x = (mdl.cdf[None, :, 1:] < u[..., None]).sum(dim=2)          # smallest s with F(s) >= u
    return torch.minimum(torch.maximum(x, lo), hi)


# --- 5. acceptance -----------------------------------------------------------------------------

def accept(x, t, chi, x_new, t_new, chi_new, generator=None):
    """Accept (x_new, t_new) with probability min(1, c(x) / c(x_new)), as log u <= sum_j log c_j(x)
    - sum_j log c_j(x_new) (Eq. 12); a rejected chain keeps (x, t, chi). Returns x, t, chi, ok."""
    log_ratio = counts(chi).double().log().sum(dim=1) - counts(chi_new).double().log().sum(dim=1)
    u = torch.rand(len(x), dtype=torch.float64, device=x.device, generator=generator)
    ok = u.log() <= log_ratio
    keep = ok[:, None]
    return (torch.where(keep, x_new, x), torch.where(keep, t_new, t),
            [torch.where(keep, cn, co) for cn, co in zip(chi_new, chi)], ok)


# --- 6. label refresh --------------------------------------------------------------------------

def refresh_labels(chi, generator=None):
    """Every label uniform over its system's boxes containing x, independently (Eq. 9)."""
    return torch.stack([_categorical(ch, len(ch), generator) for ch in chi], dim=1)


def _check_labels(chi, t):
    rows = torch.arange(len(t), device=t.device)
    for j, ch in enumerate(chi):
        assert bool(ch[rows, t[:, j]].all()), f"label of system {j} does not contain the state"


STEPS = ("order", "leave_one_out", "draw_label", "draw_state", "membership", "accept", "refresh",
         "checks")
TUPLE_STEPS = ("partners", "choose_tuple", "tuple_state", "tuple_accept", "tuple_refresh")


class _Clock:
    """Adds the wall-clock time since the previous lap to sec[name], waiting for the GPU first so
    that each step is charged its own work. Does nothing when sec is None."""

    def __init__(self, sec, device):
        self.sec, self.cuda = sec, torch.device(device).type == "cuda"
        self.t = time.perf_counter() if sec is not None else None

    def lap(self, name):
        if self.sec is None:
            return
        if self.cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.sec[name] = self.sec.get(name, 0.0) + now - self.t
        self.t = now


def sweep(mdl, x, t, chi, generator=None, check=True, sec=None, cache=None):
    """One sweep (Section 7): for each system j in a random order, steps 3a to 6.
    Returns x, t, chi and, per system (C, J), ok (jump accepted) and changed (accepted with a
    new label, lam' != lam_j). Given a dict sec, adds each step's wall-clock seconds to it
    (keys STEPS; membership is the evaluation of chi at the proposed state). With one observed
    system the label probabilities never change; a dict cache keeps them between sweeps."""
    C, J = t.shape
    ok_all = torch.zeros(C, J, dtype=torch.bool, device=x.device)
    changed = torch.zeros(C, J, dtype=torch.bool, device=x.device)
    rows = torch.arange(C, device=x.device)
    clock = _Clock(sec, x.device)
    order = system_order(J, generator, x.device).tolist()                      # 2
    clock.lap("order")
    for j in order:
        a, b = leave_one_out(mdl, t, j)                                         # 3a
        clock.lap("leave_one_out")
        logq = None
        if J == 1:                                                              # [a, b] = everything
            logq = cache.get("logq") if cache is not None else None
            if logq is None:
                logq = label_logprob(mdl, j, a[:1], b[:1])
                if cache is not None:
                    cache["logq"] = logq
        lam, log_zeta = draw_label(mdl, j, a, b, generator, logq)               # 3b
        clock.lap("draw_label")
        bx = mdl.systems[j]
        lo, hi = torch.maximum(a, bx.lo[lam]), torch.minimum(b, bx.hi[lam])
        x_new = draw_state(mdl, lo, hi, generator)                              # 4
        clock.lap("draw_state")
        chi_new = membership(mdl, x_new)
        t_new = t.clone()
        t_new[:, j] = lam
        clock.lap("membership")
        if check:
            assert bool(torch.isfinite(log_zeta).all()), "zeta_j = 0 at a jump"
            for k, ch in enumerate(chi_new):                                   # containment
                assert bool(ch[rows, t_new[:, k]].all()), "proposal outside its composed box"
            clock.lap("checks")
        old = t[:, j]
        x, t, chi, ok = accept(x, t, chi, x_new, t_new, chi_new, generator)     # 5
        ok_all[:, j], changed[:, j] = ok, ok & (lam != old)
        clock.lap("accept")
        t = refresh_labels(chi, generator)                                      # 6
        clock.lap("refresh")
        if check:
            assert bool((counts(chi) >= 1).all()), "a chain left the event"     # feasibility
            _check_labels(chi, t)                                               # label support
            clock.lap("checks")
    return x, t, chi, ok_all, changed


# --- T. tuple jump -----------------------------------------------------------------------------

class Rho(NamedTuple):
    """Partner distribution (Eq. rho): the systems in the order g_1, ..., g_J, the first label
    uniform and each later one with probability proportional to
    P(L^{g_k, lam} | L^{t_{1:k-1}})^gamma. order None draws a uniformly random order at every
    tuple jump (draw_partners)."""
    order: list
    gamma: float


def partner_rho(mdl, gamma=1.0, order=None):
    """The partner distribution of Eq. (rho): 0 < gamma <= 1, and an order of the systems that is
    either fixed (order given) or, by default (None), a uniformly random permutation drawn at
    every tuple jump without looking at the state, as the sweep's system_order. Each jump is
    then the jump of a fixed-order rho, which keeps the target, and so does their mixture.
    gamma = 1 gives the later stages of the sequential draw (Eq. 22); a smaller gamma flattens
    them (binary components with failure probability q and lower reference states: weight
    beta^d with beta = q^gamma)."""
    J = len(mdl.systems)
    if order is not None:
        order = [int(j) for j in order]
        if sorted(order) != list(range(J)):
            raise ValueError("order must be a permutation of the systems")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must lie in (0, 1]")
    return Rho(order, float(gamma))


def _stage_logprob(mdl, rho, k, a, b):
    """log rho_k(. | t_{1:k-1}) over the boxes of system g_k, (N, R), for partial tuples whose
    composed box is [a, b] (N, n): uniform at k = 0. A row with no non-empty box is all -inf."""
    j = rho.order[k]
    R = len(mdl.systems[j].lo)
    if k == 0:
        return torch.full((len(a), R), -math.log(R), dtype=torch.float64, device=a.device)
    lq = rho.gamma * label_logprob(mdl, j, a, b)
    norm = torch.logsumexp(lq, dim=1, keepdim=True)
    return torch.where(torch.isinf(norm), -torch.inf, lq - norm)


def rho_logprob(mdl, rho, tuples):
    """log rho(t) = sum_k log rho_k(t_{g_k} | t_{g_1..g_{k-1}}) for tuples (N, J), evaluated along
    each tuple in the order of rho; up to the constant of the redraw, which cancels in
    Eq. (alphatuple). rho.order must be set (draw_partners sets a random one)."""
    if rho.order is None:
        raise ValueError("rho_logprob needs the order of the systems")
    n, M = mdl.probs.shape
    N, dev = len(tuples), tuples.device
    a = torch.zeros(N, n, dtype=torch.long, device=dev)
    b = torch.full((N, n), M - 1, dtype=torch.long, device=dev)
    rows = torch.arange(N, device=dev)
    out = torch.zeros(N, dtype=torch.float64, device=dev)
    for k, j in enumerate(rho.order):
        out += _stage_logprob(mdl, rho, k, a, b)[rows, tuples[:, j]]
        bx = mdl.systems[j]
        a, b = torch.maximum(a, bx.lo[tuples[:, j]]), torch.minimum(b, bx.hi[tuples[:, j]])
    return out


def _draw_rho(mdl, rho, N, generator=None, max_try=100):
    """N tuples from rho, stage by stage in its order, and their log rho (N,). A draw that
    reaches a stage with no non-empty box is redrawn from the start, which restricts rho to
    the tuples with a non-empty box (T^+)."""
    n, M = mdl.probs.shape
    J, dev = len(mdl.systems), mdl.probs.device
    tuples = torch.zeros(N, J, dtype=torch.long, device=dev)
    log_r = torch.zeros(N, dtype=torch.float64, device=dev)
    pending = torch.ones(N, dtype=torch.bool, device=dev)
    for _ in range(max_try):
        idx = torch.nonzero(pending).squeeze(1)
        P = len(idx)
        a = torch.zeros(P, n, dtype=torch.long, device=dev)
        b = torch.full((P, n), M - 1, dtype=torch.long, device=dev)
        lam_all = torch.zeros(P, J, dtype=torch.long, device=dev)
        lr = torch.zeros(P, dtype=torch.float64, device=dev)
        alive = torch.ones(P, dtype=torch.bool, device=dev)
        for k, j in enumerate(rho.order):
            sel = torch.nonzero(alive).squeeze(1)
            if len(sel) == 0:
                break
            lp = _stage_logprob(mdl, rho, k, a[sel], b[sel])
            top = lp.max(dim=1, keepdim=True).values
            dead = torch.isinf(top[:, 0])
            alive[sel[dead]] = False
            sel, lp, top = sel[~dead], lp[~dead], top[~dead]
            lam = _categorical((lp - top).exp(), len(sel), generator)
            lam_all[sel, j] = lam
            lr[sel] += lp[torch.arange(len(sel), device=dev), lam]
            bx = mdl.systems[j]
            a[sel], b[sel] = torch.maximum(a[sel], bx.lo[lam]), torch.minimum(b[sel], bx.hi[lam])
        done = torch.nonzero(alive).squeeze(1)
        tuples[idx[done]], log_r[idx[done]] = lam_all[done], lr[done]
        pending[idx[done]] = False
        if not pending.any():
            return tuples, log_r
    raise RuntimeError(f"no partner tuple with a non-empty box for {int(pending.sum())} draws in {max_try} tries")


def _corners(mdl, tuples):
    """Corners a, b (..., n) of the composed boxes of tuples (..., J) (Eq. 5); empty where a > b."""
    n, M = mdl.probs.shape
    shape = (*tuples.shape[:-1], n)
    a = torch.zeros(shape, dtype=torch.long, device=tuples.device)
    b = torch.full(shape, M - 1, dtype=torch.long, device=tuples.device)
    for j, bx in enumerate(mdl.systems):
        a, b = torch.maximum(a, bx.lo[tuples[..., j]]), torch.minimum(b, bx.hi[tuples[..., j]])
    return a, b


def box_logprob(mdl, a, b):
    """log P(a <= X <= b) = sum_i log P(a_i <= X_i <= b_i) for boxes (..., n) (Eq. 4); -inf
    for an empty box."""
    comp = torch.arange(a.shape[-1], device=a.device).expand_as(a)
    lp = _mass(mdl.cdf, comp, a, torch.maximum(a, b)).log().sum(dim=-1)
    return torch.where((a > b).any(dim=-1), -torch.inf, lp)


def tuple_membership(chi, tau):
    """1(x in L^{tau^k}), (C, m + 1): x lies in the box of every system the tuple names."""
    inside = torch.ones(tau.shape[:2], dtype=torch.bool, device=tau.device)
    for j, ch in enumerate(chi):
        inside &= ch.gather(1, tau[..., j])
    return inside


def draw_partners(mdl, rho, t, m, generator=None):
    """T1. tau (C, m + 1, J): tau[:, 0] = t, the tuple just refreshed, and tau[:, 1:] m partner
    tuples drawn independently from rho (partner_rho; Eq. rho), redrawn at a dead end. Also
    log rho(tau^k), (C, m + 1), at every position: for the held tuple evaluated along it in the
    same order, as s_tau needs (Eq. ctau). If rho.order is None, one random order is drawn
    here (system_order) and used for the partners and the held tuple alike; it is returned as
    the third value."""
    C, J = t.shape
    if rho.order is None:
        rho = rho._replace(order=system_order(J, generator, t.device).tolist())
    partners, log_r = _draw_rho(mdl, rho, C * m, generator)
    tau = torch.cat([t[:, None], partners.view(C, m, J)], dim=1)
    log_rho_tau = torch.cat([rho_logprob(mdl, rho, t)[:, None], log_r.view(C, m)], dim=1)
    return tau, log_rho_tau, rho.order


def choose_tuple(mdl, tau, generator=None):
    """T2. k (C,) with probability P(L^{tau^k}) / sum_l P(L^{tau^l}). Positions count
    separately, so a tuple repeated in tau is chosen in proportion to how often it appears; an
    empty box has probability zero and is never chosen. Returns k, the corners lo, hi (C, n) of
    the chosen box, and log P(L^{tau^l}) (C, m + 1)."""
    a, b = _corners(mdl, tau)
    lp = box_logprob(mdl, a, b)
    top = lp.max(dim=1, keepdim=True).values
    if torch.isinf(top).any():
        raise ValueError("every tuple of tau has an empty box")
    k = _categorical((lp - top).exp(), len(tau), generator)
    rows = torch.arange(len(tau), device=tau.device)
    return k, a[rows, k], b[rows, k], lp


def accept_tuple(x, chi, x_new, chi_new, tau, log_rho_tau, generator=None):
    """T4. Accept x_new with probability
        min{1, c(x)/c(x') * c_tau(x)/c_tau(x') * s_tau(x')/s_tau(x)}          (Eq. alphatuple)
    where c_tau(x) counts the positions of tau whose box contains x and s_tau(x) sums 1 / rho
    over them (Eq. ctau); a rejected chain keeps (x, chi). The labels must be refreshed after
    (Eq. 9) either way, since the held tuple need not contain x_new. Returns x, chi, ok."""
    def log_terms(ch):
        inside = tuple_membership(ch, tau)
        log_c = counts(ch).double().log().sum(dim=1)
        log_c_tau = inside.sum(dim=1).double().log()
        log_s_tau = torch.logsumexp(torch.where(inside, -log_rho_tau, -torch.inf), dim=1)
        return log_c + log_c_tau - log_s_tau

    log_ratio = log_terms(chi) - log_terms(chi_new)
    u = torch.rand(len(x), dtype=torch.float64, device=x.device, generator=generator)
    ok = u.log() <= log_ratio
    keep = ok[:, None]
    return torch.where(keep, x_new, x), [torch.where(keep, cn, co) for cn, co in zip(chi_new, chi)], ok


def tuple_jump(mdl, x, t, chi, rho, m, generator=None, check=True, sec=None):
    """T1-T4 and the refresh (Eq. 9), from (x, t) with t just refreshed. Returns x, t, chi, ok (C,)
    and changed (C,), accepted from a tuple other than the held one. Given a dict sec, adds
    each step's seconds to it (keys TUPLE_STEPS; tuple_state draws x' and evaluates chi there)."""
    clock = _Clock(sec, x.device)
    rows = torch.arange(len(x), device=x.device)
    tau, log_rho_tau, _ = draw_partners(mdl, rho, t, m, generator)             # T1
    clock.lap("partners")
    k, lo, hi, _ = choose_tuple(mdl, tau, generator)                           # T2
    clock.lap("choose_tuple")
    x_new = draw_state(mdl, lo, hi, generator)                                 # T3
    chi_new = membership(mdl, x_new)
    clock.lap("tuple_state")
    if check:
        assert bool(tuple_membership(chi, tau)[:, 0].all()), "the held tuple does not contain the state"
        assert bool(torch.isfinite(log_rho_tau[:, 0]).all()), "rho is zero at the held tuple"
        assert bool(tuple_membership(chi_new, tau)[rows, k].all()), "proposal outside its composed box"
        clock.lap("checks")
    x, chi, ok = accept_tuple(x, chi, x_new, chi_new, tau, log_rho_tau, generator)   # T4
    changed = ok & (tau[rows, k] != t).any(dim=1)
    clock.lap("tuple_accept")
    t = refresh_labels(chi, generator)
    clock.lap("tuple_refresh")
    if check:
        assert bool((counts(chi) >= 1).all()), "a chain left the event"
        _check_labels(chi, t)
        clock.lap("checks")
    return x, t, chi, ok, changed


# --- 7. run ------------------------------------------------------------------------------------

def run(mdl, n_chain, n_sweep, n_burn=0, init=None, generator=None, check=True, timed=True,
        n_partner=0, rho=None):
    """n_sweep sweeps of C = n_chain chains, recording the state after every sweep past n_burn,
    whether or not its jumps were accepted (Section 7). init = (x, t) starts there instead of
    the sequential draw. With n_partner = m > 0, every sweep ends with a tuple jump with m
    partners from rho (by default partner_rho(mdl): a random order per jump, gamma = 1). Returns a trace over
    the T = n_sweep - n_burn recorded sweeps:
        x         : (T, C, n) uint8   states
        t         : (T, C, J) int32   labels after the sweep's last refresh
        c         : (T, C, J) int32   counts c_j(x)
        ok        : (T, C, J) bool    jump on system j accepted
        changed   : (T, C, J) bool    jump on system j accepted with a new label
        tuple_ok, tuple_changed : (T, C) bool   tuple jump accepted, and accepted from a partner
                                      tuple other than the held one (with n_partner > 0)
        sweep_sec : (n_sweep,)        wall-clock seconds of every sweep, burn-in included
        sec       : seconds of the start ("init") and, with timed, of each step of the sweeps
                    (STEPS, TUPLE_STEPS), summed over all n_sweep sweeps; timing waits for the
                    GPU after every step, which costs a little
        p_event   : estimate of P(E) from the sequential draw (None with init)"""
    dev = mdl.probs.device
    if mdl.probs.shape[1] > 256:
        raise ValueError("states are recorded as uint8")

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    sec = {"init": 0.0, **{k: 0.0 for k in STEPS + TUPLE_STEPS}}
    if n_partner and rho is None:
        rho = partner_rho(mdl)
    t0 = time.perf_counter()
    if init is None:
        x, t, chi, p_event = initialise(mdl, n_chain, generator)
    else:
        x, t = init[0].long().to(dev), init[1].long().to(dev)
        chi, p_event = membership(mdl, x), None
        assert bool((counts(chi) >= 1).all()), "a starting state lies outside the event"
        _check_labels(chi, t)
    sync()
    sec["init"] = time.perf_counter() - t0
    rec = {k: [] for k in ("x", "t", "c", "ok", "changed") + (("tuple_ok", "tuple_changed") if n_partner else ())}
    sweep_sec = torch.zeros(n_sweep, dtype=torch.float64)
    cache = {}
    for s in range(n_sweep):
        t0 = time.perf_counter()
        x, t, chi, ok, changed = sweep(mdl, x, t, chi, generator, check, sec if timed else None, cache)
        if n_partner:
            x, t, chi, t_ok, t_changed = tuple_jump(mdl, x, t, chi, rho, n_partner, generator, check,
                                                    sec if timed else None)
        sync()
        sweep_sec[s] = time.perf_counter() - t0
        if s >= n_burn:
            rec["x"].append(x.to(torch.uint8))
            rec["t"].append(t.int())
            rec["c"].append(counts(chi).int())
            rec["ok"].append(ok)
            rec["changed"].append(changed)
            if n_partner:
                rec["tuple_ok"].append(t_ok)
                rec["tuple_changed"].append(t_changed)
    trace = {k: torch.stack(v) for k, v in rec.items()}
    return {**trace, "p_event": p_event, "sec": sec, "sweep_sec": sweep_sec, "n_sweep": n_sweep,
            "n_burn": n_burn}


# --- 8-12. reports -----------------------------------------------------------------------------

def acceptance_rate(trace):
    """8. Fraction of jumps accepted, per system, (J,)."""
    return trace["ok"].double().mean(dim=(0, 1))


def label_change_rate(trace):
    """9. Fraction of accepted jumps whose new label differs from the old, per system, (J,).
    Near zero means the chain is not moving between that system's boxes."""
    return trace["changed"].double().sum(dim=(0, 1)) / trace["ok"].double().sum(dim=(0, 1)).clamp(min=1)


def tuple_rates(trace):
    """8-9 for the tuple jump: the fraction of tuple jumps accepted, and the fraction of accepted
    ones that came from a partner tuple other than the held one."""
    ok = trace["tuple_ok"].double()
    return ok.mean().item(), trace["tuple_changed"].double().sum().item() / max(ok.sum().item(), 1.0)


def occupancy(trace, mdl):
    """10. Frequency of each reference box of each system in the recorded labels, a list of
    (R_j,); at stationarity box lam of system j has sum_x pi(x) chi_{j,lam}(x) / c_j(x) (Eq. 10).
    A box never occupied marks a region the chains have not visited."""
    t = trace["t"].long()
    return [torch.bincount(t[..., j].flatten(), minlength=len(bx.lo)).double() / t[..., j].numel()
            for j, bx in enumerate(mdl.systems)]


def mixing(trace, n_state, n_batch=20, chunk=8):
    """11. For every indicator f = 1(X_i = s), (n, M) each:
        mean  : pooled posterior mean
        rho1  : lag-1 autocorrelation, centred per chain and pooled
        tau   : integrated autocorrelation time by batch means (n_batch batches per chain)
        ess   : C T / tau
        rhat  : between-chain agreement (Gelman-Rubin), about 1 when the chains agree
    A constant indicator gets tau = 1 and rhat = 1; one that is constant within each chain but
    differs between chains gets rhat = inf."""
    x = trace["x"]
    T, C, n = x.shape
    L = T // n_batch
    T = L * n_batch
    x = x[:T]
    out = {k: torch.zeros(n, n_state, dtype=torch.float64) for k in ("mean", "rho1", "tau", "ess", "rhat")}
    for s in range(n_state):
        for i0 in range(0, n, chunk):
            f = (x[:, :, i0:i0 + chunk] == s).double()                          # (T, C, k)
            m_c = f.mean(dim=0)                                                 # chain means (C, k)
            z = f - m_c
            W = (z ** 2).sum(dim=0).mean(dim=0) / (T - 1)                       # within-chain variance
            lag = (z[1:] * z[:-1]).sum(dim=(0, 1))
            bm = f.view(n_batch, L, C, -1).mean(dim=1)                          # batch means
            s2_bm = L * ((bm - m_c) ** 2).sum(dim=0).mean(dim=0) / (n_batch - 1)
            B = T * m_c.var(dim=0) if C > 1 else torch.zeros_like(W)
            var_plus = (T - 1) / T * W + B / T
            const = W == 0
            rho1 = torch.where(const, 0.0, lag / (z ** 2).sum(dim=(0, 1)).clamp(min=1e-300))
            tau = torch.where(const, 1.0, (s2_bm / W.clamp(min=1e-300)).clamp(min=1e-12))
            rhat = torch.where(const, torch.where(B > 0, torch.inf, 1.0),
                               (var_plus / W.clamp(min=1e-300)).sqrt())
            sl = slice(i0, i0 + chunk)
            out["mean"][sl, s] = m_c.mean(dim=0).cpu()
            out["rho1"][sl, s] = rho1.cpu()
            out["tau"][sl, s] = tau.cpu()
            out["ess"][sl, s] = (C * T / tau).cpu()
            out["rhat"][sl, s] = rhat.cpu()
    return out


def mean_counts(trace):
    """12. Chain averages of c(x) = prod_j c_j(x), which estimates E_pi[c] (kappa when J = 1,
    Eq. 15), and of each c_j(x), (J,)."""
    c = trace["c"].double()
    return c.prod(dim=-1).mean().item(), c.mean(dim=(0, 1))


def timing(trace):
    """13. Milliseconds per sweep, for all C chains together: each step of the scheme (STEPS,
    summed over the systems of a sweep, and TUPLE_STEPS of the tuple jump, zero without one),
    their total, and the median of the measured sweep times; plus the start ("init", ms in
    all). Shows where the time goes before the ensemble size is fixed (Section 12, profiling)."""
    n = trace["n_sweep"]
    ms = {k: 1e3 * trace["sec"][k] / n for k in STEPS + TUPLE_STEPS}
    return {**ms, "steps_total": sum(ms.values()), "sweep_median": 1e3 * trace["sweep_sec"].median().item(),
            "init": 1e3 * trace["sec"]["init"]}
