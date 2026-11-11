"""Tests of rsr.mcmc on the demonstrations of "Box jumps under component independence".

Network of Figure 1: route 1 is X1 then X2, route 2 is X3, binary components with P(X_i = 0) = 0.1.
    Demonstration 1  route 1 failed            boxes (0,1,1), (1,0,1)
    Demonstration 2  route 2 failed            box (1,1,0)
    Demonstration 3  both routes failed
    Demonstration 4  route 1 failed, route 2 surviving (upper box (0,0,1))
    Demonstration 5  three-state components, systems A and B failed, X2 shared
Two tests per function of the scheme; exact values come from enumerating every state."""

import pytest
import torch

from rsr import mcmc

P01 = torch.tensor([[0.1, 0.9]] * 3, dtype=torch.float64)                       # Eq. (19)
P5 = torch.tensor([[0.2, 0.3, 0.5], [0.05, 0.15, 0.80], [0.05, 0.15, 0.80]], dtype=torch.float64)


def lower(*ls):
    """Failure boxes [0, l] from lower reference states l."""
    hi = torch.tensor(ls)
    return torch.zeros_like(hi), hi


def upper(*us, M=2):
    """Survival boxes [u, M - 1] from upper reference states u."""
    lo = torch.tensor(us)
    return lo, torch.full_like(lo, M - 1)


ROUTE1_FAILS = lower((0, 1, 1), (1, 0, 1))                                     # Eq. (20)
ROUTE2_FAILS = lower((1, 1, 0))
ROUTE2_SURVIVES = upper((0, 0, 1))
DEMO1 = mcmc.model([ROUTE1_FAILS], P01)
DEMO2 = mcmc.model([ROUTE2_FAILS], P01)
DEMO3 = mcmc.model([ROUTE1_FAILS, ROUTE2_FAILS], P01)
DEMO4 = mcmc.model([ROUTE1_FAILS, ROUTE2_SURVIVES], P01)
# Demonstration 4 with a third failure box of route 1 that needs x3 = 0, which the survival of
# route 2 rules out (the case discussed after Demonstration 4): same event, an empty tuple.
DEMO4_EMPTY = mcmc.model([lower((0, 1, 1), (1, 0, 1), (0, 1, 0)), ROUTE2_SURVIVES], P01)
DEMO5 = mcmc.model([lower((0, 2, 2), (1, 0, 2)), lower((2, 0, 1), (2, 1, 0))], P5)
A1, A2, B1, B2 = 0, 1, 0, 1                                                    # Demonstration 5 labels


def gen():
    return torch.Generator().manual_seed(0)


def exact(mdl):
    """Every state (S, n), its target pi (S,), memberships and counts (S, J), and P(E)."""
    n, M = mdl.probs.shape
    states = torch.cartesian_prod(*[torch.arange(M)] * n)
    p = mdl.probs[torch.arange(n), states].prod(dim=1)
    chi = mcmc.membership(mdl, states)
    c = mcmc.counts(chi)
    w = p * (c >= 1).all(dim=1)
    return {"states": states, "p": p, "pi": w / w.sum(), "chi": chi, "c": c, "P_E": w.sum().item()}


def frequency(x, M):
    """Frequency of every state in samples (..., n), in the order of exact()["states"]."""
    n = x.shape[-1]
    code = (x.long() * M ** torch.arange(n - 1, -1, -1)).sum(dim=-1).flatten()
    return torch.bincount(code, minlength=M ** n).double() / len(code)


def in_event(mdl, x):
    return (mcmc.counts(mcmc.membership(mdl, x.reshape(-1, x.shape[-1]).long())) >= 1).all()


def exact_jump_rates(mdl):
    """One observed system: the stationary acceptance rate of a jump and the fraction of accepted
    jumps that change the label. The jump draws (lam', x') with probability p(x') chi_lam'(x') / Z."""
    ex = exact(mdl)
    chi, c, pi = ex["chi"][0].double(), ex["c"][:, 0].double(), ex["pi"]
    inside = c > 0
    chi, c, pi, p = chi[inside], c[inside], pi[inside], ex["p"][inside]
    Q = (p[:, None] * chi).T / (p[:, None] * chi).sum()                         # (R, S'): (lam', x')
    acc = torch.minimum(torch.ones(1, dtype=torch.float64), c[:, None] / c[None, :])   # (S, S')
    lab = chi / c[:, None]                                                      # pi(lam | x)
    ok = pi @ acc @ Q.sum(dim=0)
    changed = sum(pi[s] * lab[s, l] * (Q[lp] * acc[s]).sum()
                  for s in range(len(pi)) for l in range(chi.shape[1]) for lp in range(chi.shape[1]) if lp != l)
    return ok.item(), (changed / ok).item()


# --- model -------------------------------------------------------------------------------------

def test_model_reads_rsr_masks_as_the_same_boxes():
    masks = torch.tensor([[[1, 0], [1, 1], [1, 1]], [[1, 1], [1, 0], [1, 1]]], dtype=torch.int32)
    from_masks, from_corners = mcmc.model([masks], P01).systems[0], DEMO1.systems[0]
    for a, b in zip(from_masks, from_corners):
        assert torch.equal(a, b)


def test_model_rejects_a_mask_that_is_not_an_interval():
    masks = torch.tensor([[[1, 0, 1], [1, 1, 1], [1, 1, 1]]], dtype=torch.int32)
    with pytest.raises(ValueError):
        mcmc.model([masks], P5)


# --- 1. initialise -----------------------------------------------------------------------------

def test_initialise_starts_every_chain_in_the_event_with_labels_that_contain_it():
    # Demonstration 4, and its variant whose third failure box is empty with route 2 surviving.
    for mdl in (DEMO4, DEMO4_EMPTY):
        x, t, chi, _ = mcmc.initialise(mdl, 5000, gen())
        assert in_event(mdl, x)
        rows = torch.arange(len(x))
        assert all(ch[rows, t[:, j]].all() for j, ch in enumerate(chi))


def test_initialise_weights_estimate_the_event_probability():
    # Lemma 2: E[prod zeta / c(x)] = P(E): 0.019 (Dem. 3), 0.171 (Dem. 4, also with the empty
    # tuple, whose draws restart and count as weight 0), 0.0065 (Dem. 5).
    for mdl, p_e in ((DEMO3, 0.019), (DEMO4, 0.171), (DEMO4_EMPTY, 0.171), (DEMO5, 0.0065)):
        assert abs(exact(mdl)["P_E"] - p_e) < 1e-12
        p_hat = mcmc.initialise(mdl, 200_000, gen())[3]
        assert abs(p_hat / p_e - 1) < 0.02, (p_e, p_hat)


# --- 2. system_order ---------------------------------------------------------------------------

def test_system_order_is_a_permutation_of_the_observed_systems():
    g = gen()
    for J in (len(DEMO1.systems), len(DEMO3.systems), 3):
        for _ in range(20):
            assert sorted(mcmc.system_order(J, g).tolist()) == list(range(J))


def test_system_order_takes_both_orders_of_two_systems_equally_often():
    # Demonstration 3 has two systems: each order with probability 1/2.
    g = gen()
    first = torch.tensor([mcmc.system_order(2, g)[0].item() for _ in range(20_000)])
    assert abs(first.double().mean() - 0.5) < 0.015


# --- 3a. leave_one_out -------------------------------------------------------------------------

def test_leave_one_out_is_the_box_of_the_other_systems():
    # Demonstration 4, t = (1, 1): system 2 holds {x3 = 1}; system 1 holds [0, (0, 1, 1)].
    t = torch.tensor([[0, 0]])
    a, b = mcmc.leave_one_out(DEMO4, t, 0)
    assert a.tolist() == [[0, 0, 1]] and b.tolist() == [[1, 1, 1]]
    a, b = mcmc.leave_one_out(DEMO4, t, 1)
    assert a.tolist() == [[0, 0, 0]] and b.tolist() == [[0, 1, 1]]


def test_leave_one_out_is_the_whole_space_for_one_system_and_a_slice_for_two():
    # Demonstration 1 (J = 1): the whole space. Demonstration 5 with lam_B = B2: [0, (2, 1, 0)].
    a, b = mcmc.leave_one_out(DEMO1, torch.tensor([[1]]), 0)
    assert a.tolist() == [[0, 0, 0]] and b.tolist() == [[1, 1, 1]]
    a, b = mcmc.leave_one_out(DEMO5, torch.tensor([[A1, B2]]), 0)
    assert a.tolist() == [[0, 0, 0]] and b.tolist() == [[2, 1, 0]]


# --- 3b. draw_label ----------------------------------------------------------------------------

def test_draw_label_weights_boxes_by_their_probability_within_the_slice():
    # Demonstration 5 trace: lam_B = B2 held -> A2 w.p. 0.00125 / 0.00325 = 0.385; lam_A = A2
    # held -> B1 w.p. 0.8; lam_B = B1 held -> A1 w.p. 0.002 / 0.007 = 0.286 (Eq. 13).
    for t, j, box, prob, zeta in (([A1, B2], 0, A2, 0.125 / 0.325, 0.325),
                                  ([A2, B1], 1, B1, 0.8, 0.25),
                                  ([A2, B1], 0, A1, 0.2 / 0.7, 0.7)):
        t = torch.tensor([t] * 50_000)
        lam, log_zeta = mcmc.draw_label(DEMO5, j, *mcmc.leave_one_out(DEMO5, t, j), gen())
        assert torch.allclose(log_zeta.exp(), torch.tensor(zeta, dtype=torch.float64))
        assert abs((lam == box).double().mean() - prob) < 0.01


def test_draw_label_never_proposes_a_box_that_misses_the_slice():
    # Demonstration 4: with route 2 surviving (x3 = 1) both failure boxes of route 1 have
    # conditional probability 0.1, so each is drawn w.p. 1/2; a third box needing x3 = 0 has
    # probability zero and is never drawn.
    for mdl, R in ((DEMO4, 2), (DEMO4_EMPTY, 3)):
        a, b = mcmc.leave_one_out(mdl, torch.zeros(50_000, 2, dtype=torch.long), 0)
        q = mcmc.label_logprob(mdl, 0, a[:1], b[:1]).exp()[0]
        assert torch.allclose(q[:2], torch.tensor([0.1, 0.1], dtype=torch.float64))
        lam, _ = mcmc.draw_label(mdl, 0, a, b, gen())
        freq = torch.bincount(lam, minlength=R).double() / len(lam)
        assert abs(freq[0] - 0.5) < 0.01 and (R == 2 or freq[2] == 0)


# --- 4. draw_state -----------------------------------------------------------------------------

def test_draw_state_fixes_constrained_components_and_draws_the_rest_from_the_prior():
    # Demonstration 4, t = 0: the box L^(2,1) = [(0,0,1), (1,0,1)] fixes x2 = 0 and x3 = 1 and
    # draws x1 from its prior on {0, 1}: P(x1 = 0) = 0.1.
    lo, hi = torch.tensor([[0, 0, 1]] * 50_000), torch.tensor([[1, 0, 1]] * 50_000)
    x = mcmc.draw_state(DEMO4, lo, hi, gen())
    assert (x[:, 1] == 0).all() and (x[:, 2] == 1).all()
    assert abs((x[:, 0] == 0).double().mean() - 0.1) < 0.005


def test_draw_state_truncates_three_state_components():
    # Demonstration 5, box (A2, B1) = [0, (1, 0, 1)]: x1 in {0, 1} w.p. (0.2, 0.3) / 0.5,
    # x2 = 0, x3 in {0, 1} w.p. (0.05, 0.15) / 0.2.
    x = mcmc.draw_state(DEMO5, torch.zeros(100_000, 3, dtype=torch.long), torch.tensor([[1, 0, 1]] * 100_000), gen())
    assert (x <= torch.tensor([1, 0, 1])).all()
    assert abs((x[:, 0] == 0).double().mean() - 0.4) < 0.005
    assert abs((x[:, 2] == 0).double().mean() - 0.25) < 0.005


# --- 5. accept ---------------------------------------------------------------------------------

def jump_from(mdl, x, t, x_new, t_new, n):
    x, x_new = torch.tensor([x] * n), torch.tensor([x_new] * n)
    t, t_new = torch.tensor([t] * n), torch.tensor([t_new] * n)
    return x, t, mcmc.membership(mdl, x), x_new, t_new, mcmc.membership(mdl, x_new)


def test_accept_halves_moves_into_a_doubly_covered_state():
    # Demonstration 1 trace: from (1,0,1) (c = 1) to (0,0,1) (c = 2) w.p. 1/2; to (1,0,0)
    # (c = 1) always.
    args = jump_from(DEMO1, (1, 0, 1), [1], (0, 0, 1), [0], 50_000)
    _, _, _, ok = mcmc.accept(*args, generator=gen())
    assert abs(ok.double().mean() - 0.5) < 0.01
    args = jump_from(DEMO1, (1, 0, 1), [1], (1, 0, 0), [1], 1000)
    assert mcmc.accept(*args, generator=gen())[3].all()


def test_accept_rejection_keeps_the_state_labels_and_memberships():
    # Demonstration 5 trace, t = 2: from (1,0,1) (c = 1) to (0,0,0) (c = 4) w.p. 1/4.
    args = jump_from(DEMO5, (1, 0, 1), [A2, B1], (0, 0, 0), [A1, B1], 50_000)
    x, t, chi, ok = mcmc.accept(*args, generator=gen())
    assert abs(ok.double().mean() - 0.25) < 0.01
    assert (x[~ok] == torch.tensor([1, 0, 1])).all() and (t[~ok] == torch.tensor([A2, B1])).all()
    assert (x[ok] == 0).all() and (t[ok] == torch.tensor([A1, B1])).all()
    for ch, ch_old, ch_new in zip(chi, args[2], args[5]):
        assert torch.equal(ch[~ok], ch_old[~ok]) and torch.equal(ch[ok], ch_new[ok])


# --- 6. refresh_labels -------------------------------------------------------------------------

def test_refresh_labels_is_uniform_over_the_tuples_containing_the_state():
    # Demonstration 5: (0,0,0) lies in all four composed boxes, each drawn w.p. 1/4.
    chi = mcmc.membership(DEMO5, torch.zeros(80_000, 3, dtype=torch.long))
    t = mcmc.refresh_labels(chi, gen())
    freq = torch.bincount(t[:, 0] * 2 + t[:, 1], minlength=4).double() / len(t)
    assert (freq - 0.25).abs().max() < 0.01


def test_refresh_labels_keeps_a_system_whose_state_lies_in_one_box():
    # Demonstration 5 trace, after t = 0: at (1,0,0) lam_A = A2, the only box of A containing it,
    # and lam_B is uniform over {B1, B2}.
    chi = mcmc.membership(DEMO5, torch.tensor([[1, 0, 0]] * 50_000))
    t = mcmc.refresh_labels(chi, gen())
    assert (t[:, 0] == A2).all()
    assert abs((t[:, 1] == B1).double().mean() - 0.5) < 0.01


# --- 7. run ------------------------------------------------------------------------------------

def test_run_records_every_sweep_inside_the_event():
    # Demonstration 4: one record per sweep after burn-in, accepted or not, every state in E and
    # every recorded label containing it.
    tr = mcmc.run(DEMO4, n_chain=500, n_sweep=30, n_burn=10, generator=gen())
    assert tr["x"].shape == (20, 500, 3) and tr["t"].shape == tr["ok"].shape == (20, 500, 2)
    assert (~tr["ok"]).any()
    x, t = tr["x"].reshape(-1, 3).long(), tr["t"].reshape(-1, 2).long()
    chi = mcmc.membership(DEMO4, x)
    rows = torch.arange(len(x))
    assert in_event(DEMO4, x) and all(ch[rows, t[:, j]].all() for j, ch in enumerate(chi))


def test_run_samples_the_exact_target():
    # Demonstrations 3, 4 and 5: total variation distance to the enumerated target (Eqs. 21, 56).
    for mdl, M in ((DEMO3, 2), (DEMO4, 2), (DEMO5, 3)):
        tr = mcmc.run(mdl, n_chain=4000, n_sweep=60, n_burn=10, generator=gen())
        tv = 0.5 * (frequency(tr["x"], M) - exact(mdl)["pi"]).abs().sum()
        assert tv < 0.01, tv


# --- 8. acceptance_rate ------------------------------------------------------------------------

def test_acceptance_rate_matches_the_stationary_rate_of_demonstration_1():
    # Proposals land in the overlap (0,0,x3) w.p. 0.1 and are then accepted w.p. 1/2 unless
    # the chain is already there: about 0.95.
    ok_exact, _ = exact_jump_rates(DEMO1)
    tr = mcmc.run(DEMO1, n_chain=4000, n_sweep=60, n_burn=10, generator=gen())
    assert abs(mcmc.acceptance_rate(tr).item() - ok_exact) < 0.005
    assert 0.94 < ok_exact < 0.96


def test_acceptance_rate_is_one_without_overlap_and_per_system_with_two():
    # Demonstration 2: one box, c = 1 everywhere, every jump accepted. Demonstration 5: about
    # 0.76 for each system (Section 9).
    assert mcmc.acceptance_rate(mcmc.run(DEMO2, 200, 20, generator=gen())).item() == 1.0
    rate = mcmc.acceptance_rate(mcmc.run(DEMO5, 4000, 60, 10, generator=gen()))
    assert rate.shape == (2,) and ((rate - 0.76).abs() < 0.02).all()


# --- 9. label_change_rate ----------------------------------------------------------------------

def test_label_change_rate_is_zero_for_a_system_with_one_box():
    # Demonstration 2, and route 2 in Demonstration 3: one reference state, so no label change.
    assert mcmc.label_change_rate(mcmc.run(DEMO2, 200, 20, generator=gen())).item() == 0.0
    rate = mcmc.label_change_rate(mcmc.run(DEMO3, 2000, 40, 10, generator=gen()))
    assert rate[1] == 0.0 and rate[0] > 0.3


def test_label_change_rate_matches_demonstration_1():
    # The jump proposes either box w.p. 1/2, so about half of the accepted jumps change it.
    _, changed_exact = exact_jump_rates(DEMO1)
    tr = mcmc.run(DEMO1, n_chain=4000, n_sweep=60, n_burn=10, generator=gen())
    assert abs(mcmc.label_change_rate(tr).item() - changed_exact) < 0.005


# --- 10. occupancy -----------------------------------------------------------------------------

def exact_occupancy(mdl):
    """pi(lam_j = r) = sum_x pi(x) chi_{j,r}(x) / c_j(x) per system (Eq. 10)."""
    ex = exact(mdl)
    return [(ex["pi"][:, None] * ch.double() / ex["c"][:, j:j + 1].clamp(min=1)).sum(dim=0)
            for j, ch in enumerate(ex["chi"])]


def test_occupancy_is_even_for_the_two_symmetric_boxes_of_demonstration_1():
    occ = mcmc.occupancy(mcmc.run(DEMO1, 4000, 60, 10, generator=gen()), DEMO1)
    assert torch.allclose(exact_occupancy(DEMO1)[0], torch.tensor([0.5, 0.5], dtype=torch.float64))
    assert (occ[0] - 0.5).abs().max() < 0.005


def test_occupancy_matches_the_label_marginal_of_demonstration_5():
    # lam_A = A1 w.p. 0.0385 + 0.1154 + 0.2308 = 0.385: the overlap states split their mass.
    occ = mcmc.occupancy(mcmc.run(DEMO5, 4000, 60, 10, generator=gen()), DEMO5)
    ex = exact_occupancy(DEMO5)
    assert abs(ex[0][A1] - 5 / 13) < 1e-9
    assert all((o - e).abs().max() < 0.01 for o, e in zip(occ, ex))


# --- 11. mixing --------------------------------------------------------------------------------

def test_mixing_of_independent_draws_in_demonstration_2():
    # One box: every jump is an independent draw from p(. | L^1), so rho1 = 0, tau = 1 and
    # ESS = C T; x3 = 0 always, a constant indicator.
    mx = mcmc.mixing(mcmc.run(DEMO2, 400, 400, generator=gen()), 2)
    free = mx["rho1"][:2]
    assert free.abs().max() < 0.02 and ((mx["ess"][:2] / (400 * 400)) - 1).abs().max() < 0.15
    assert (mx["rhat"][:2] < 1.01).all()
    assert mx["tau"][2].eq(1).all() and mx["rhat"][2].eq(1).all()


def test_mixing_detects_chains_held_in_different_failure_modes():
    # Demonstration 3: chains that never leave L^(1,1) (x1 = 0) and chains that never leave
    # L^(2,1) (x2 = 0) disagree on P(x1 = 0); the sweep's chains agree.
    T, C = 200, 200
    lo = torch.zeros(T * C, 3, dtype=torch.long)
    hi = torch.tensor([[0, 1, 0]] * (T * C // 2) + [[1, 0, 0]] * (T * C // 2))
    x = mcmc.draw_state(DEMO3, lo, hi.view(C, T, 3).transpose(0, 1).reshape(-1, 3), gen())
    stuck = mcmc.mixing({"x": x.view(T, C, 3).to(torch.uint8)}, 2)
    assert stuck["rhat"][0, 0] > 2
    good = mcmc.mixing(mcmc.run(DEMO3, C, T, 20, generator=gen()), 2)
    assert good["rhat"][:2].max() < 1.05


# --- 12. mean_counts ---------------------------------------------------------------------------

def test_mean_counts_estimates_kappa_for_one_system():
    # Demonstration 1: E_pi[c] = kappa = Z / P(E) = 0.2 / 0.19 (Eq. 15).
    c, c_j = mcmc.mean_counts(mcmc.run(DEMO1, 4000, 60, 10, generator=gen()))
    assert abs(c - 0.2 / 0.19) < 0.003 and abs(c_j.item() - c) < 1e-12


def test_mean_counts_of_demonstration_5():
    # E_pi[c] = 0.0769 * 4 + (0.2308 + 0.1154) * 2 + 0.2308 + 0.3462 = 1.577, and per system.
    ex = exact(DEMO5)
    c_exact = (ex["pi"] * ex["c"].prod(dim=1)).sum()
    cj_exact = (ex["pi"][:, None] * ex["c"]).sum(dim=0)
    assert abs(c_exact - 1.577) < 0.001
    c, c_j = mcmc.mean_counts(mcmc.run(DEMO5, 4000, 60, 10, generator=gen()))
    assert abs(c - c_exact) < 0.01 and (c_j - cj_exact).abs().max() < 0.01


# --- 13. timing --------------------------------------------------------------------------------

def test_timing_charges_every_step_of_the_sweep():
    # Demonstration 5: every step takes time, and the steps add up to no more than the sweeps.
    tr = mcmc.run(DEMO5, n_chain=1000, n_sweep=20, n_burn=5, generator=gen())
    ms = mcmc.timing(tr)
    assert tr["sweep_sec"].shape == (20,)
    assert all(ms[k] > 0 for k in mcmc.STEPS)
    assert ms["steps_total"] <= 1e3 * tr["sweep_sec"].sum().item() / 20 + 1e-9
    assert ms["init"] > 0


def test_timing_can_be_switched_off_but_sweeps_are_still_timed():
    # Demonstration 1 without per-step timing and without assertions: the steps read zero.
    tr = mcmc.run(DEMO1, n_chain=1000, n_sweep=20, generator=gen(), check=False, timed=False)
    ms = mcmc.timing(tr)
    assert ms["steps_total"] == 0 and ms["sweep_median"] > 0


# --- T1. draw_partners -------------------------------------------------------------------------

def tuple_code(tau):
    return tau[..., 0] * 2 + tau[..., 1]                                       # Demonstration 5: A, B


RHO5_AB = [0.25, 0.25, 0.4, 0.1]                           # Demonstration 5, gamma = 1, order (A, B)
RHO5_BA = [1 / 7, 4 / 13, 5 / 14, 5 / 26]                  # order (B, A); tuples (A1B1, A1B2, A2B1, A2B2)


def test_draw_partners_keeps_the_held_tuple_and_draws_partners_by_eq_rho():
    # Demonstration 5, order (A, B): lam_A uniform, then lam_B by P(L^{B,lam} | L^{A,lam_A})^gamma:
    # given A1 both B boxes have 0.01 -> 1/2 each; given A2, B1 has 0.2 and B2 0.05 -> 0.8, 0.2
    # (gamma = 1) or sqrt: 2/3, 1/3 (gamma = 1/2). Order (B, A): lam_B uniform; given B1, A1 and
    # A2 have 0.2 and 0.5 -> 2/7, 5/7; given B2, 0.2 and 0.125 -> 8/13, 5/13 (Section 9's 0.385).
    t = torch.tensor([[A2, B1]] * 10_000)
    for gamma, order, want in ((1.0, (0, 1), RHO5_AB), (0.5, (0, 1), [0.25, 0.25, 1 / 3, 1 / 6]),
                               (1.0, (1, 0), RHO5_BA)):
        rho = mcmc.partner_rho(DEMO5, gamma, order)
        tau, log_rho_tau, used = mcmc.draw_partners(DEMO5, rho, t, 8, gen())
        assert tau.shape == (10_000, 9, 2) and torch.equal(tau[:, 0], t) and used == list(order)
        freq = torch.bincount(tuple_code(tau[:, 1:]).flatten(), minlength=4).double() / (10_000 * 8)
        assert (freq - torch.tensor(want, dtype=torch.float64)).abs().max() < 0.01, (gamma, freq)
        assert torch.allclose(log_rho_tau.exp(), torch.tensor(want, dtype=torch.float64)[tuple_code(tau)])


def test_draw_partners_with_a_random_order_per_jump():
    # Demonstration 5, order None: each call draws one order, uniformly, and uses it for the
    # partners and the held tuple alike, so log rho matches that order's table; over calls the
    # partners follow the average of the two tables.
    rho, g = mcmc.partner_rho(DEMO5), gen()
    t = torch.tensor([[A2, B1]] * 200)
    tables = {(0, 1): torch.tensor(RHO5_AB, dtype=torch.float64), (1, 0): torch.tensor(RHO5_BA, dtype=torch.float64)}
    freq, n_ab = torch.zeros(4, dtype=torch.float64), 0
    for _ in range(1000):
        tau, log_rho_tau, used = mcmc.draw_partners(DEMO5, rho, t, 4, g)
        assert torch.allclose(log_rho_tau.exp(), tables[tuple(used)][tuple_code(tau)])
        freq += torch.bincount(tuple_code(tau[:, 1:]).flatten(), minlength=4).double()
        n_ab += used == [0, 1]
    assert abs(n_ab / 1000 - 0.5) < 0.05
    assert (freq / freq.sum() - (tables[(0, 1)] + tables[(1, 0)]) / 2).abs().max() < 0.01


def test_draw_partners_redraws_dead_ends_and_leaves_them_empty_after_max_try():
    # Demonstration 4 with the third failure box (needs x3 = 0), order (route 1, route 2): a
    # draw of that box finds no survival box of route 2 left. With 100 attempts it is redrawn,
    # so partners are (1, 1) or (2, 1), each w.p. 1/2, and rho = 1/3 at every tuple, the held one
    # included (the restriction's constant cancels in the acceptance). With one attempt, 1/3 of
    # the partners are left empty (log rho = -inf) and the rest are (1, 1) or (2, 1) as before.
    t = torch.tensor([[0, 0]] * 5000 + [[1, 0]] * 5000)
    third = torch.tensor(1 / 3, dtype=torch.float64).log()
    tau, log_rho_tau, _ = mcmc.draw_partners(DEMO4_EMPTY, mcmc.partner_rho(DEMO4_EMPTY, order=(0, 1), max_try=100),
                                             t, 6, gen())
    freq = torch.bincount(tau[:, 1:, 0].flatten(), minlength=3).double() / (10_000 * 6)
    assert freq[2] == 0 and abs(freq[0] - 0.5) < 0.01 and (tau[..., 1] == 0).all()
    assert torch.allclose(log_rho_tau, third)
    tau, log_rho_tau, _ = mcmc.draw_partners(DEMO4_EMPTY, mcmc.partner_rho(DEMO4_EMPTY, order=(0, 1), max_try=1),
                                             t, 6, gen())
    empty = torch.isinf(log_rho_tau)
    assert not empty[:, 0].any() and abs(empty[:, 1:].double().mean() - 1 / 3) < 0.01
    assert torch.allclose(log_rho_tau[~empty], third) and (tau[:, 1:, 0][~empty[:, 1:]] != 2).all()
    for bad in ({"gamma": 0.0}, {"max_try": 0}):
        with pytest.raises(ValueError):
            mcmc.partner_rho(DEMO4_EMPTY, **bad)


# --- T2. choose_tuple --------------------------------------------------------------------------

def test_choose_tuple_weights_tuples_by_their_box_probability():
    # Demonstration 5, tau = all four tuples: P(L^t) = 0.002 (A1,B1), 0.002 (A1,B2), 0.005 (A2,B1),
    # 0.00125 (A2,B2), so (A2,B1) is chosen w.p. 0.005 / 0.01025 = 0.488.
    tau = torch.tensor([[[A1, B1], [A1, B2], [A2, B1], [A2, B2]]] * 50_000)
    k, lo, hi, lp = mcmc.choose_tuple(DEMO5, tau, gen())
    box = torch.tensor([0.002, 0.002, 0.005, 0.00125], dtype=torch.float64)
    assert torch.allclose(lp[0].exp(), box)
    freq = torch.bincount(k, minlength=4).double() / len(k)
    assert (freq - box / box.sum()).abs().max() < 0.01
    sel = k == 2
    assert (lo[sel] == 0).all() and (hi[sel] == torch.tensor([1, 0, 1])).all()


def test_choose_tuple_counts_repeats_by_position_and_never_picks_an_empty_box():
    # Demonstration 4 with the third failure box: (1, 1) and (2, 1) both have P = 0.09; (3, 1) is
    # empty. tau = ((1,1), (1,1), (2,1), (3,1)): positions 0, 1, 2 w.p. 1/3 each, position 3 never.
    # An empty partner (live False) is skipped too: with position 1 empty, 0 and 2 w.p. 1/2.
    tau = torch.tensor([[[0, 0], [0, 0], [1, 0], [2, 0]]] * 30_000)
    k, _, _, lp = mcmc.choose_tuple(DEMO4_EMPTY, tau, gen())
    assert torch.allclose(lp[0, :3].exp(), torch.tensor(0.09, dtype=torch.float64)) and lp[0, 3] == -torch.inf
    freq = torch.bincount(k, minlength=4).double() / len(k)
    assert freq[3] == 0 and (freq[:3] - 1 / 3).abs().max() < 0.01
    live = torch.tensor([[True, False, True, True]] * 30_000)
    k, _, _, _ = mcmc.choose_tuple(DEMO4_EMPTY, tau, gen(), live)
    freq = torch.bincount(k, minlength=4).double() / len(k)
    assert freq[1] == 0 and freq[3] == 0 and abs(freq[0] - 0.5) < 0.01


# --- T4. accept_tuple --------------------------------------------------------------------------

def tuple_move(mdl, x, x_new, tau, n):
    x, x_new = torch.tensor([x] * n), torch.tensor([x_new] * n)
    tau = torch.tensor([tau] * n)
    return x, mcmc.membership(mdl, x), x_new, mcmc.membership(mdl, x_new), tau


def test_accept_tuple_reduces_to_the_rho_ratio_in_the_typical_case():
    # Demonstration 5: (1,0,1) lies only in (A2,B1) and (0,1,0) only in (A1,B2), c = 1 at both;
    # rho = 0.4 and 0.25 (order (A, B), gamma = 1), so the move to (0,1,0) is always accepted
    # and the move back w.p. 0.25 / 0.4 = 0.625.
    rho = mcmc.partner_rho(DEMO5, order=(0, 1))
    for x, x_new, tau, want in (((1, 0, 1), (0, 1, 0), [[A2, B1], [A1, B2]], 1.0),
                                ((0, 1, 0), (1, 0, 1), [[A1, B2], [A2, B1]], 0.625)):
        x, chi, x_new, chi_new, tau = tuple_move(DEMO5, x, x_new, tau, 50_000)
        log_rho_tau = mcmc.rho_logprob(DEMO5, rho, tau.view(-1, 2)).view(len(tau), -1)
        _, _, ok = mcmc.accept_tuple(x, chi, x_new, chi_new, tau, log_rho_tau, gen())
        assert abs(ok.double().mean() - want) < 0.01


def test_accept_tuple_corrects_for_overlap_and_keeps_the_state_on_rejection():
    # rho by Eq. (rho) throughout.
    # Demonstration 1 (one system, so rho is uniform): from (1,0,1) (box 2 only) to (0,0,1)
    # (both boxes, c = 2), tau = (2, 1): 1/2 * 1/2 * 2 = 1/2.
    # Demonstration 5: from (1,0,1) (only in (A2,B1), c = 1) to (0,0,1) (in (A1,B1) and (A2,B1),
    # c = 2), tau = ((A2,B1), (A1,B1)). Order (A, B): rho = 0.4 and 0.25, s(x) = 2.5,
    # s(x') = 6.5, so 1/2 * 1/2 * 6.5 / 2.5 = 0.65. Order (B, A): given B1, A1 and A2 have 0.2
    # and 0.5, so rho = 5/14 and 1/7, s(x) = 2.8, s(x') = 9.8, and 1/4 * 9.8 / 2.8 = 0.875.
    for mdl, order, tau, want in ((DEMO1, (0,), [[1], [0]], 0.5),
                                  (DEMO5, (0, 1), [[A2, B1], [A1, B1]], 0.65),
                                  (DEMO5, (1, 0), [[A2, B1], [A1, B1]], 0.875)):
        x, chi, x_new, chi_new, tau = tuple_move(mdl, (1, 0, 1), (0, 0, 1), tau, 50_000)
        J = tau.shape[2]
        log_rho_tau = mcmc.rho_logprob(mdl, mcmc.partner_rho(mdl, 1.0, order), tau.view(-1, J)).view(len(tau), -1)
        x_out, chi_out, ok = mcmc.accept_tuple(x, chi, x_new, chi_new, tau, log_rho_tau, gen())
        assert abs(ok.double().mean() - want) < 0.01, (order, ok.double().mean())
        assert torch.equal(x_out[~ok], x[~ok]) and torch.equal(x_out[ok], x_new[ok])
        for c_out, c_old, c_new in zip(chi_out, chi, chi_new):
            assert torch.equal(c_out[~ok], c_old[~ok]) and torch.equal(c_out[ok], c_new[ok])


# --- tuple_jump --------------------------------------------------------------------------------

def test_tuple_jump_alone_keeps_the_exact_target():
    # Proposition (tuple): T1-T4 and the refresh, without the sweep, sample the enumerated target
    # of Demonstrations 3, 4 (with the empty tuple) and 5, for gamma = 1 and 1/2, both orders and
    # a random one, and with one attempt per partner (dead ends left empty).
    for mdl, M in ((DEMO3, 2), (DEMO4_EMPTY, 2), (DEMO5, 3)):
        for gamma, order, max_try in ((1.0, (0, 1), 10), (0.5, (1, 0), 10), (1.0, None, 10), (1.0, (0, 1), 1)):
            rho = mcmc.partner_rho(mdl, gamma, order, max_try)
            g = gen()
            x, t, chi, _ = mcmc.initialise(mdl, 4000, g)
            xs = []
            for s in range(60):
                x, t, chi, ok, _ = mcmc.tuple_jump(mdl, x, t, chi, rho, 3, g)
                if s >= 10:
                    xs.append(x)
            tv = 0.5 * (frequency(torch.stack(xs), M) - exact(mdl)["pi"]).abs().sum()
            assert tv < 0.01, (gamma, order, max_try, tv)


def test_run_with_tuple_jumps_records_and_times_them():
    # Demonstration 5: every sweep followed by a tuple jump with 4 partners; the target is still
    # exact, the tuple jumps are recorded (some accepted from a partner) and timed.
    tr = mcmc.run(DEMO5, n_chain=4000, n_sweep=60, n_burn=10, generator=gen(), n_partner=4)
    assert tr["tuple_ok"].shape == tr["tuple_changed"].shape == (50, 4000)
    acc, changed = mcmc.tuple_rates(tr)
    assert 0 < acc < 1 and 0 < changed < 1
    assert 0.5 * (frequency(tr["x"], 3) - exact(DEMO5)["pi"]).abs().sum() < 0.01
    ms = mcmc.timing(tr)
    assert all(ms[k] > 0 for k in mcmc.TUPLE_STEPS)


# --- observe, with_probs, start, step: one step at a time, by observed system -----------------

RULES = {"route1": (upper((1, 1, 0)), ROUTE1_FAILS),                     # (survival, failure) reference states
         "route2": (ROUTE2_SURVIVES, ROUTE2_FAILS)}


def test_observe_picks_failure_or_survival_boxes_by_observed_state():
    # Route 1 failed and route 2 surviving is Demonstration 4; route 2 failed alone, Demonstration 2.
    mdl = mcmc.observe(RULES, {"route1": 0, "route2": 1}, P01)
    assert mdl.names == ["route1", "route2"]
    for got, want in zip(mdl.systems, DEMO4.systems):
        assert torch.equal(got.lo, want.lo) and torch.equal(got.hi, want.hi)
    mdl = mcmc.observe(RULES, {"route2": 0}, P01)
    assert mdl.names == ["route2"] and torch.equal(mdl.systems[0].hi, DEMO2.systems[0].hi)


def test_observe_rejects_unknown_systems_states_and_empty_boxes():
    with pytest.raises(KeyError):
        mcmc.observe(RULES, {"route3": 0}, P01)
    with pytest.raises(ValueError):
        mcmc.observe(RULES, {"route1": 2}, P01)
    empty = torch.zeros(0, 3, 2, dtype=torch.int32)                       # no failure reference states
    with pytest.raises(ValueError):
        mcmc.observe({"route1": (upper((1, 1, 0)), empty)}, {"route1": 0}, P01)


def test_with_probs_keeps_the_boxes_and_checks_the_prior():
    mdl = mcmc.with_probs(DEMO4, torch.tensor([[0.6, 0.4]] * 3))
    assert all(a is b for a, b in zip(mdl.systems, DEMO4.systems))
    assert torch.allclose(mdl.cdf[:, 1], torch.tensor(0.6, dtype=torch.float64))
    with pytest.raises(ValueError):
        mcmc.with_probs(DEMO4, torch.tensor([[0.5, 0.5]] * 2))
    per_chain = mcmc.with_probs(DEMO4, P01.expand(10, -1, -1))
    with pytest.raises(ValueError):
        mcmc.initialise(per_chain, 20)                                    # prior for 10 chains


def test_with_probs_gives_each_chain_its_own_prior():
    # Half the chains with P(X_i = 0) = 0.1, half with 0.6. Unconstrained draws follow each
    # chain's prior; in Demonstration 4 (x3 = 1 held) box (0,1,1) has conditional probability
    # P(x1 = 0) = 0.1 or 0.6.
    C = 40_000
    q = torch.tensor([[0.1, 0.9]] * 3 + [[0.6, 0.4]] * 3, dtype=torch.float64).view(2, 3, 2).repeat_interleave(C // 2, 0)
    mdl = mcmc.with_probs(DEMO4, q)
    x = mcmc.draw_state(mdl, torch.zeros(C, 3, dtype=torch.long), torch.ones(C, 3, dtype=torch.long), gen())
    half = C // 2
    assert abs((x[:half] == 0).double().mean() - 0.1) < 0.01 and abs((x[half:] == 0).double().mean() - 0.6) < 0.01
    a, b = mcmc.leave_one_out(mdl, torch.zeros(C, 2, dtype=torch.long), 0)
    p = mcmc.label_logprob(mdl, 0, a, b).exp()[:, 0]
    assert torch.allclose(p[:half], torch.tensor(0.1, dtype=torch.float64))
    assert torch.allclose(p[half:], torch.tensor(0.6, dtype=torch.float64))


def test_start_draws_feasible_chains_and_from_the_prior_without_observations():
    mdl = mcmc.observe(RULES, {"route1": 0, "route2": 1}, P01)
    x, t, chi = mcmc.start(mdl, 2000, gen())
    rows = torch.arange(len(x))
    assert in_event(mdl, x) and all(ch[rows, t[:, j]].all() for j, ch in enumerate(chi))
    state = mcmc.start(mcmc.observe(RULES, {}, P01), 50_000, gen())       # nothing observed
    assert state.t.shape == (50_000, 0) and abs((state.x == 0).double().mean() - 0.1) < 0.005


def test_start_keeps_a_warm_start_where_it_lies_in_the_event():
    # Demonstration 3 (both routes failed): (0,1,0) is in E and is kept, (1,1,1) is not and is
    # replaced by a sequential draw.
    x0 = torch.tensor([[0, 1, 0], [1, 1, 1]] * 500)
    x, t, chi = mcmc.start(DEMO3, 1000, gen(), x0=x0)
    assert (x[0::2] == torch.tensor([0, 1, 0])).all()
    assert in_event(DEMO3, x[1::2]) and not (x[1::2] == 1).all(dim=1).any()
    rows = torch.arange(len(x))
    assert all(ch[rows, t[:, j]].all() for j, ch in enumerate(chi))


def test_step_is_exact_inside_a_gibbs_sampler_whose_prior_changes():
    # A latent z per chain, P(z = 1) = 0.3, with P(X_i = 0 | z) = 0.1 or 0.4, and both routes
    # observed failed (Demonstration 3). Gibbs: z | x exactly, then x | z by one step under the
    # chain's prior (with_probs), with and without tuple jumps. The pairs (z, x) follow the
    # enumerated joint p(z) p(x | z) 1(x in E) / P(E).
    q0 = torch.tensor([0.1, 0.4], dtype=torch.float64)                     # P(X_i = 0 | z)
    pz = torch.tensor([0.7, 0.3], dtype=torch.float64)
    states = torch.cartesian_prod(*[torch.arange(2)] * 3)
    in_e = (mcmc.counts(mcmc.membership(DEMO3, states)) >= 1).all(dim=1)
    lik = torch.where(states[None] == 0, q0[:, None, None], 1 - q0[:, None, None]).prod(dim=2)   # (2, 8)
    joint = pz[:, None] * lik * in_e
    joint = (joint / joint.sum()).flatten()                               # z major, then x
    mdl = mcmc.observe(RULES, {"route1": 0, "route2": 0}, P01)
    for n_partner in (0, 2):
        g, C = gen(), 4000
        z = (torch.rand(C, generator=g) < 0.3).long()
        state = mcmc.start(mcmc.with_probs(mdl, torch.stack([q0[z], 1 - q0[z]], -1)[:, None].expand(C, 3, 2)), C, g)
        codes = []
        for s in range(60):
            code = (state.x * torch.tensor([4, 2, 1])).sum(dim=1)
            post = pz[:, None] * lik[:, code]                              # z | x
            z = (torch.rand(C, generator=g) < post[1] / post.sum(dim=0)).long()
            q = torch.stack([q0[z], 1 - q0[z]], -1)[:, None].expand(C, 3, 2)
            state, info = mcmc.step(mcmc.with_probs(mdl, q), state, g, n_partner=n_partner)
            if s >= 10:
                codes.append(z * 8 + (state.x * torch.tensor([4, 2, 1])).sum(dim=1))
        freq = torch.bincount(torch.cat(codes), minlength=16).double() / (50 * C)
        assert 0.5 * (freq - joint).abs().sum() < 0.015, (n_partner, freq, joint)


def test_step_reports_its_moves_and_redraws_without_observations():
    mdl = mcmc.observe(RULES, {"route1": 0, "route2": 1}, P01)
    g = gen()
    state = mcmc.start(mdl, 500, g)
    state, info = mcmc.step(mdl, state, g, n_partner=3)
    assert info["ok"].shape == info["changed"].shape == (500, 2) and info["tuple_ok"].shape == (500,)
    rows = torch.arange(500)
    assert in_event(mdl, state.x) and all(ch[rows, state.t[:, j]].all() for j, ch in enumerate(state.chi))
    free = mcmc.observe(RULES, {}, P01)                                    # nothing observed
    state = mcmc.start(free, 50_000, g)
    new, info = mcmc.step(free, state, g)
    assert info["tuple_ok"] is None and not torch.equal(new.x, state.x)
    assert abs((new.x == 0).double().mean() - 0.1) < 0.005
