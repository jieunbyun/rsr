"""Certificate-augmented MCMC over reference states (single layer).

States and reference states use rsr's format. Each row of ``x`` is one chain,
so every step processes all chains at once.

    refs  : (R, n, M) 0/1 masks, refs[r, i, s] = 1 if box r allows X_i = s
    probs : (n, M)    independent prior, probs[i, s] = P(X_i = s)
    x     : (C, n, M) one-hot states of C chains
    c     : (C,)      overlap counts c(x) = number of boxes containing x

One step: x_new, gamma = proposal(x, ...); x, c, ok = accept(x_new, x, c, ...).
run() repeats this and returns the traces; gamma and ok feed the monitoring functions.

Several observed systems: x, gamma, c = init_multi(refs_list, ...), then
x, gamma, c, ok = step_multi(x, gamma, c, refs_list, ...) per sweep; gamma, c, ok are (C, S).
"""

import time

import torch

from rsr.rsr import classify_samples_with_indices


def _membership(x, refs):
    """chi[c, r] = 1(x_c in box r), Eq. (4), by rsr's classify_samples_with_indices.
    The subset check is the same for upper and lower reference states."""
    res = classify_samples_with_indices(x, refs, [], return_violations=True)
    return res["violations_upper"] == 0                        # (C, R)


def _corner(box, upper):
    """One-hot corner of each box (C, n, M): the lowest allowed state per component
    for upper reference states, the highest for lower ones."""
    M = box.shape[2]
    corner = box.argmax(dim=2) if upper else M - 1 - box.flip(2).argmax(dim=2)
    return torch.nn.functional.one_hot(corner, M).int()


def _draw(w, generator=None):
    """x ~ w componentwise, for weights w (C, n, M); returns one-hot (C, n, M)."""
    draw = torch.multinomial(w.reshape(-1, w.shape[2]).double(), 1, generator=generator)
    return torch.nn.functional.one_hot(draw.reshape(w.shape[:2]), w.shape[2]).int()


def init(refs, n_chain, upper=False):
    """Start chain k at the corner of box k mod R (Eq. 58)."""
    x = _corner(refs[torch.arange(n_chain) % len(refs)].int(), upper)
    return x, _membership(x, refs).sum(dim=1)


def proposal(x, refs, probs, generator=None):
    """Draw gamma uniformly from the boxes containing x (C.1),
    then x' ~ p(. | x' in box gamma) componentwise (Kernel I, Eq. 27).
    Returns x' and gamma (C,), the box index each chain proposed from."""
    chi = _membership(x, refs)
    gamma = torch.multinomial(chi.double(), 1, generator=generator).squeeze(1)
    return _draw(probs[None, :, :] * refs[gamma], generator), gamma


def accept(x_new, x, c, refs, u=None, generator=None):
    """Accept x_new with probability min(1, c(x) / c(x_new)) (Eq. 11);
    a rejected chain keeps its current state. Returns x, c and ok (C,) accept flags."""
    c_new = _membership(x_new, refs).sum(dim=1)
    if u is None:
        u = torch.rand(len(x), dtype=torch.float64, generator=generator)
    ok = u < (c / c_new).clamp(max=1.0)
    return torch.where(ok[:, None, None], x_new, x), torch.where(ok, c_new, c), ok


# --- several observed systems: refs_list holds each system's reference states ---

def _intersection(refs_list, gamma):
    """The box each chain is held in: the intersection of every system's current box, (C, n, M)."""
    box = refs_list[0][gamma[:, 0]].int()
    for s in range(1, len(refs_list)):
        box = box * refs_list[s][gamma[:, s]]
    return box


def init_multi(refs_list, n_chain, upper=False):
    """Start chain k at the corner of the intersection of box k mod R_s of every system s.
    Returns x, gamma (C, S) and c (C, S), c[:, s] = number of system s's boxes containing x."""
    gamma = torch.stack([torch.arange(n_chain) % len(r) for r in refs_list], dim=1)
    box = _intersection(refs_list, gamma)
    if (box.sum(dim=2) == 0).any():
        raise ValueError("the starting boxes of the systems do not intersect")
    x = _corner(box, upper)
    return x, gamma, torch.stack([_membership(x, r).sum(dim=1) for r in refs_list], dim=1)


def step_multi(x, gamma, c, refs_list, probs, generator=None):
    """One sweep over the systems. For each system s in turn: redraw gamma_s uniformly from
    its boxes containing x (C.1), propose x' ~ p(. | x' in every system's current box), and
    accept with min(1, prod_s c_s(x) / c_s(x')). With one system this is proposal + accept.
    Returns x, gamma, c and ok (C, S), whether the move for each system was accepted."""
    gamma, oks = gamma.clone(), []
    for s, refs in enumerate(refs_list):
        chi = _membership(x, refs)
        gamma[:, s] = torch.multinomial(chi.double(), 1, generator=generator).squeeze(1)
        x_new = _draw(probs[None, :, :] * _intersection(refs_list, gamma), generator)

        c_new = torch.stack([_membership(x_new, r).sum(dim=1) for r in refs_list], dim=1)
        u = torch.rand(len(x), dtype=torch.float64, generator=generator)
        ok = u < (c.prod(dim=1) / c_new.prod(dim=1)).clamp(max=1.0)
        x, c = torch.where(ok[:, None, None], x_new, x), torch.where(ok[:, None], c_new, c)
        oks.append(ok)
    return x, gamma, c, torch.stack(oks, dim=1)


def run(refs, probs, n_chain, n_step, n_burn=0, n_thin=1, upper=False, generator=None):
    """Run n_chain chains for n_step steps. After the first n_burn steps (burn-in),
    keep every n_thin-th step only, to save memory; check n_thin with autocorrelation().
    Returns traces over the T kept steps:
        x     : (T, C, n, M) chain states after each kept step (the samples for inference)
        c     : (T, C)       overlap counts of those states
        gamma : (T, C)       box each kept step proposed from
        ok    : (T, C)       whether each kept step accepted
        sec   : {"init", "proposal", "accept"} total wall-clock seconds, over all steps
    gamma and ok are a subsample of all steps, so their monitors stay unbiased."""
    sec = {"init": 0.0, "proposal": 0.0, "accept": 0.0}

    def timed(name, f, *args, **kwargs):
        if refs.is_cuda:
            torch.cuda.synchronize()                           # GPU calls return before they finish
        t0 = time.perf_counter()
        out = f(*args, **kwargs)
        if refs.is_cuda:
            torch.cuda.synchronize()
        sec[name] += time.perf_counter() - t0
        return out

    x, c = timed("init", init, refs, n_chain, upper)
    kept = []
    for t in range(n_step):
        x_new, gamma = timed("proposal", proposal, x, refs, probs, generator)
        x, c, ok = timed("accept", accept, x_new, x, c, refs, generator=generator)
        if t >= n_burn and (t - n_burn) % n_thin == 0:
            kept.append((x, c, gamma, ok))
    return (*(torch.stack(v) for v in zip(*kept)), sec)


# --- monitoring: traces are stacked per step, shape (T, C) ---

def acceptance_rate(ok):
    """Fraction of steps each chain accepted its proposal, (C,)."""
    return ok.double().mean(dim=0)


def label_frequency(gamma, n_ref):
    """freq[c, r] = fraction of steps chain c proposed from box r, (C, R).
    A mixed chain approaches pi(gamma = r), proportional to sum over x in box r of p(x) / c(x)."""
    return torch.nn.functional.one_hot(gamma, n_ref).double().mean(dim=0)


def autocorrelation(trace, max_lag=5):
    """rho[k-1] = lag-k autocorrelation of a scalar trace (T, C), k = 1..max_lag,
    pooled over chains. Lags count kept samples, so if rho[0] is far from 0 on a
    thinned run, a wider n_thin is needed."""
    z = trace.double() - trace.double().mean()
    return torch.stack([(z[:-k] * z[k:]).mean() for k in range(1, max_lag + 1)]) / (z * z).mean()
