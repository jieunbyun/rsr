"""Two chains on the network of Figure 1, with the system's own reference states.
Reference states and samples are (n_comp, n_state) 0/1 matrices, as in rsr."""

import torch

from rsr import mcmc

PROBS = torch.tensor([[0.1, 0.9]] * 3, dtype=torch.float64)    # Eq. (35)
LOWER = torch.tensor([
    [[1, 0], [1, 1], [1, 0]],                                  # x1 <= 0 and x3 <= 0
    [[1, 1], [1, 0], [1, 0]],                                  # x2 <= 0 and x3 <= 0
], dtype=torch.int32)
UPPER = torch.tensor([
    [[0, 1], [0, 1], [1, 1]],                                  # x1 >= 1 and x2 >= 1
    [[1, 1], [1, 1], [0, 1]],                                  # x3 >= 1
], dtype=torch.int32)


def onehot(*chains):
    """onehot((0, 1, 0), (1, 0, 0)) -> (2, 3, 2) samples, one row per chain."""
    return torch.nn.functional.one_hot(torch.tensor(chains), 2).int()


def test_init_failure_starts_each_chain_at_a_lower_reference_state():
    x, c = mcmc.init(LOWER, n_chain=2)
    assert torch.equal(x, onehot((0, 1, 0), (1, 0, 0)))
    assert torch.equal(c, torch.tensor([1, 1]))


def test_init_survival_starts_each_chain_at_an_upper_reference_state():
    x, c = mcmc.init(UPPER, n_chain=2, upper=True)
    assert torch.equal(x, onehot((1, 1, 0), (0, 0, 1)))
    assert torch.equal(c, torch.tensor([1, 1]))


def test_proposal_failure_fixes_the_reference_and_redraws_the_free_component():
    # Chain 1 at (0,1,0) lies only in box 1; chain 2 at (1,0,0) only in box 2.
    n = 50_000
    x = onehot((0, 1, 0), (1, 0, 0)).repeat(n, 1, 1)
    x_new, gamma = mcmc.proposal(x, LOWER, PROBS, torch.Generator().manual_seed(0))
    states = x_new.argmax(dim=2)
    chain1, chain2 = states[0::2], states[1::2]
    assert torch.equal(gamma, torch.tensor([0, 1]).repeat(n))    # each chain's only box

    assert (chain1[:, [0, 2]] == 0).all()
    assert (chain2[:, [1, 2]] == 0).all()
    assert abs((chain1[:, 1] == 0).double().mean() - 0.1) < 0.005    # p_i(0) = 0.1
    assert abs((chain2[:, 0] == 0).double().mean() - 0.1) < 0.005


def test_proposal_survival_fixes_the_reference_and_redraws_the_free_components():
    # Chain 1 at (1,1,0) lies only in box 1; chain 2 at (0,0,1) only in box 2.
    n = 50_000
    x = onehot((1, 1, 0), (0, 0, 1)).repeat(n, 1, 1)
    x_new, gamma = mcmc.proposal(x, UPPER, PROBS, torch.Generator().manual_seed(0))
    states = x_new.argmax(dim=2)
    chain1, chain2 = states[0::2], states[1::2]
    assert torch.equal(gamma, torch.tensor([0, 1]).repeat(n))    # each chain's only box

    assert (chain1[:, [0, 1]] == 1).all()
    assert (chain2[:, 2] == 1).all()
    for redrawn in (chain1[:, 2], chain2[:, 0], chain2[:, 1]):
        assert abs((redrawn == 0).double().mean() - 0.1) < 0.005


def test_accept_failure_demo3_trace_steps_0_and_2():
    # t=0: (0,1,0), c=1 -> (0,0,0), c'=2, alpha=1/2, u=0.41: accept
    # t=2: (1,0,0), c=1 -> (0,0,0), c'=2, alpha=1/2, u=0.73: reject, state recorded again
    x = onehot((0, 1, 0), (1, 0, 0))
    x_new = onehot((0, 0, 0), (0, 0, 0))
    u = torch.tensor([0.41, 0.73], dtype=torch.float64)
    x_next, c_next, ok = mcmc.accept(x_new, x, torch.tensor([1, 1]), LOWER, u)
    assert torch.equal(x_next, onehot((0, 0, 0), (1, 0, 0)))
    assert torch.equal(c_next, torch.tensor([2, 1]))
    assert torch.equal(ok, torch.tensor([True, False]))


def test_accept_survival_into_and_out_of_the_overlap():
    # (1,1,1) is the only state in both survival boxes, so c = 2 there.
    # chain 1: (1,1,0), c=1 -> (1,1,1), c'=2, alpha=1/2, u=0.73: reject
    # chain 2: (1,1,1), c=2 -> (0,0,1), c'=1, alpha=1,   u=0.73: accept
    x = onehot((1, 1, 0), (1, 1, 1))
    x_new = onehot((1, 1, 1), (0, 0, 1))
    u = torch.tensor([0.73, 0.73], dtype=torch.float64)
    x_next, c_next, ok = mcmc.accept(x_new, x, torch.tensor([1, 2]), UPPER, u)
    assert torch.equal(x_next, onehot((1, 1, 0), (0, 0, 1)))
    assert torch.equal(c_next, torch.tensor([1, 1]))
    assert torch.equal(ok, torch.tensor([False, True]))


def run(refs, upper, n_thin=5):
    """10,000 chains, 5 burn-in steps (chains start at corners with c = 1), 15 kept samples."""
    return mcmc.run(refs, PROBS, n_chain=10_000, n_step=5 + 15 * n_thin, n_burn=5,
                    n_thin=n_thin, upper=upper, generator=torch.Generator().manual_seed(0))


def state_frequency(x):
    """Fraction of samples in each of the 8 states, indexed by the binary code x1 x2 x3."""
    code = (x.argmax(dim=3) * torch.tensor([4, 2, 1])).sum(dim=2).flatten()
    return torch.bincount(code, minlength=8).double() / len(code)


# Exact values on Figure 1. At stationarity the chain is at (x, gamma) with
# pi(x, gamma) = p(x) / (c(x) P(union)) for x in box gamma: x ~ p restricted to the
# union, and gamma uniform over the c(x) boxes containing x.

def test_acceptance_rate_failure():
    # Long-run acceptance = sum over (x, gamma) of pi(x, gamma) * a(x, gamma), where
    # a(x, gamma) = sum over x' in box gamma of p(x' | box gamma) * min(1, c(x) / c(x')).
    #
    # Boxes: box 1 = {000, 010}, box 2 = {000, 100}; 000 is in both, so c(000) = 2.
    # Prior: p(000) = 0.1^3 = 0.001, p(010) = p(100) = 0.1 * 0.9 * 0.1 = 0.009.
    # P(union) = 0.001 + 0.009 + 0.009 = 0.019.
    #
    # pi(x, gamma) = p(x) / (c(x) P(union)):
    #   pi(010, 1) = pi(100, 2) = 0.009 / 0.019       = 9/19
    #   pi(000, 1) = pi(000, 2) = 0.001 / (2 * 0.019) = 1/38   (sum of all four = 1)
    #
    # Proposal from box 1: p(000 | box 1) = 0.001 / 0.010 = 0.1, p(010 | box 1) = 0.9.
    #   a(010, 1): propose 010 (0.9) ratio 1/1 -> accept; 000 (0.1) ratio 1/2
    #              = 0.9 * 1 + 0.1 * 1/2 = 0.95
    #   a(000, 1): c = 2, every ratio 2/c' >= 1 -> always accept = 1
    # Box 2 is the mirror image, hence the factor 2:
    #   exact = 2 * (9/19 * 0.95 + 1/38 * 1) = 18.1 / 19 = 0.9526
    _, _, _, ok, _ = run(LOWER, upper=False)
    exact = 2 * (9 / 19 * 0.95 + 1 / 38)                        # = 0.9526
    assert abs(mcmc.acceptance_rate(ok).mean() - exact) < 0.005


def test_acceptance_rate_survival():
    # Long-run acceptance = sum over (x, gamma) of pi(x, gamma) * a(x, gamma), as in the failure test.
    #
    # Boxes: box 1 = {110, 111}, box 2 = {001, 011, 101, 111}; 111 is in both, so c(111) = 2.
    # Prior: p(110) = 0.9 * 0.9 * 0.1 = 0.081, p(111) = 0.9^3 = 0.729,
    #        p(001) = 0.1 * 0.1 * 0.9 = 0.009, p(011) = p(101) = 0.1 * 0.9 * 0.9 = 0.081.
    # P(union) = 0.081 + 0.729 + 0.009 + 0.081 + 0.081 = 0.981.
    #
    # pi(x, gamma) = p(x) / (c(x) P(union)):
    #   pi(110, 1)                         = 0.081 / 0.981
    #   pi(001, 2) + pi(011, 2) + pi(101, 2) = (0.009 + 0.081 + 0.081) / 0.981 = 0.171 / 0.981
    #   pi(111, 1) = pi(111, 2)            = 0.729 / (2 * 0.981) = 0.3645 / 0.981
    #
    # Proposal from box 1: P(box 1) = 0.81, p(111 | box 1) = 0.729 / 0.81 = 0.9, p(110 | box 1) = 0.1.
    #   a(110, 1): propose 110 (0.1) ratio 1/1 -> accept; 111 (0.9) ratio 1/2
    #              = 0.1 * 1 + 0.9 * 1/2 = 0.55
    # Proposal from box 2: P(box 2) = 0.9, p(111 | box 2) = 0.729 / 0.9 = 0.81, the other three 0.19.
    #   a(x, 2) for the c=1 states: 0.19 * 1 + 0.81 * 1/2 = 0.595 (same for all three)
    # a(111, 1) = a(111, 2) = 1: c = 2, every ratio 2/c' >= 1.
    # The boxes are not mirror images, so each is summed separately:
    #   exact = (0.081 * 0.55 + 0.171 * 0.595 + 2 * 0.3645 * 1) / 0.981
    #         = (0.04455 + 0.10175 + 0.729) / 0.981 = 0.8753 / 0.981 = 0.8922
    _, _, _, ok, _ = run(UPPER, upper=True)
    exact = (0.081 * 0.55 + 0.171 * 0.595 + 2 * 0.3645) / 0.981   # = 0.8922
    assert abs(mcmc.acceptance_rate(ok).mean() - exact) < 0.005


def test_label_frequency_failure():
    # Long-run label frequency pi(gamma = r) = sum over x in box r of pi(x, r):
    # the share of time the chain proposes from box r.
    #
    # Boxes: box 1 = {000, 010}, box 2 = {000, 100}; c(000) = 2.
    # From the acceptance test: pi(010, 1) = pi(100, 2) = 9/19, pi(000, 1) = pi(000, 2) = 1/38.
    #   pi(gamma = 1) = pi(010, 1) + pi(000, 1) = 9/19 + 1/38 = 19/38 = 0.5
    #   pi(gamma = 2) = pi(100, 2) + pi(000, 2) = 9/19 + 1/38 = 19/38 = 0.5   (mirror image)
    #
    # gamma only switches boxes through 000, which has pi = 1/19, so it averages slowly: atol 0.01.
    _, _, gamma, _, _ = run(LOWER, upper=False)
    exact = torch.tensor([9 / 19 + 1 / 38, 9 / 19 + 1 / 38], dtype=torch.float64)   # = 0.5, 0.5
    assert torch.allclose(mcmc.label_frequency(gamma, 2).mean(dim=0), exact, atol=0.01)


def test_label_frequency_survival():
    # Long-run label frequency pi(gamma = r) = sum over x in box r of pi(x, r), as in the failure test.
    #
    # Boxes: box 1 = {110, 111}, box 2 = {001, 011, 101, 111}; c(111) = 2, P(union) = 0.981.
    # The overlap 111 (p = 0.729) is split equally, 0.3645 to each box; the rest stays in its own box.
    #   pi(gamma = 1) = (p(110) + 0.729/2) / 0.981 = (0.081 + 0.3645) / 0.981 = 0.4455 / 0.981 = 0.4541
    #   pi(gamma = 2) = (p(001) + p(011) + p(101) + 0.729/2) / 0.981
    #                 = (0.009 + 0.081 + 0.081 + 0.3645) / 0.981 = 0.5355 / 0.981 = 0.5459
    # The two add up to 1. Box 2 gets more time because its c=1 states carry 0.171 against 0.081.
    _, _, gamma, _, _ = run(UPPER, upper=True)
    exact = torch.tensor([0.081 + 0.3645, 0.171 + 0.3645], dtype=torch.float64) / 0.981  # = 0.4541, 0.5459
    assert torch.allclose(mcmc.label_frequency(gamma, 2).mean(dim=0), exact, atol=0.005)


def test_run_failure_samples_follow_p_given_failure():
    # Target pi(x) = p(x) / P(union) on box 1 + box 2 = {000, 010, 100}, P(union) = 0.019:
    #   pi(000) = 0.001 / 0.019 = 1/19 = 0.0526,  pi(010) = pi(100) = 0.009 / 0.019 = 9/19 = 0.4737
    # Chains switch between 010 and 100 only through 000, so the split averages slowly: atol 0.01.
    x, _, _, _, _ = run(LOWER, upper=False)
    exact = torch.tensor([1, 0, 9, 0, 9, 0, 0, 0], dtype=torch.float64) / 19
    assert torch.allclose(state_frequency(x), exact, atol=0.01)


def test_run_survival_samples_follow_p_given_survival():
    # Target pi(x) = p(x) / P(union) on box 1 + box 2 = {001, 011, 101, 110, 111}, P(union) = 0.981:
    #   pi(001) = 0.009 / 0.981 = 0.0092,  pi(011) = pi(101) = pi(110) = 0.081 / 0.981 = 0.0826,
    #   pi(111) = 0.729 / 0.981 = 0.7431
    x, _, _, _, _ = run(UPPER, upper=True)
    exact = torch.tensor([0, 0.009, 0, 0.081, 0, 0.081, 0.081, 0.729], dtype=torch.float64) / 0.981
    assert torch.allclose(state_frequency(x), exact, atol=0.005)


# Exact autocorrelations on the failure case. x moves on {A = 010, B = 100, O = 000}:
#   From A (only in box 1): propose O w.p. 0.1, accept 1/2 -> A->O 0.05, A->A 0.95 (B likewise).
#   From O (c = 2, gamma = 1 or 2 w.p. 1/2): propose O w.p. 0.1, else A or B; always accept
#     -> O->O 0.1, O->A 0.45, O->B 0.45.
# A trace f(X_t) has rho_k = sum over eigenvalues lambda of w_lambda * lambda^k, where w_lambda is
# the share of Var(f) in that eigenvector; the eigenvalues besides 1 are 0.05 and 0.95.

def test_autocorrelation_failure_c_mixes_fast():
    # c = 1 + 1(O): from A or B the chain enters O w.p. 0.05 and leaves O w.p. 0.9, so c is a
    # two-state chain with lambda = 1 - 0.05 - 0.9 = 0.05 and rho_k = 0.05^k.
    # Unthinned (n_thin = 1): rho_1 = 0.05, rho_2 = 0.0025.
    _, c, _, _, _ = run(LOWER, upper=False, n_thin=1)
    exact = torch.tensor([0.05, 0.05 ** 2], dtype=torch.float64)
    assert torch.allclose(mcmc.autocorrelation(c, max_lag=2), exact, atol=0.01)


def test_autocorrelation_failure_x1_needs_a_wider_slice():
    # x1 = 1(B) = (1(A) + 1(B)) / 2 + (1(B) - 1(A)) / 2.
    #   (1(B) - 1(A)) / 2: lambda = 0.95 (A and B each stay w.p. 0.95; O maps to 0.45 - 0.45 = 0),
    #     variance 1/4 * pi(A or B) = 1/4 * 18/19 = 85.5/361.
    #   (1(A) + 1(B)) / 2 = (1 - 1(O)) / 2: lambda = 0.05, variance 1/4 * (1/19)(18/19) = 4.5/361.
    #   Var(x1) = (9/19)(10/19) = 90/361, so rho_k = 0.95 * 0.95^k + 0.05 * 0.05^k.
    # With slice 5, kept lag j is step lag 5j:
    #   rho_1 = 0.95^6 + 0.05^6 = 0.7351,  rho_2 = 0.95^11 + 0.05^11 = 0.5688
    # i.e. slice 5 is far too narrow here: chains hop between 010 and 100 only through 000.
    x, _, _, _, _ = run(LOWER, upper=False)
    x1 = x[:, :, 0, 1]                                          # (T, C), 1 if x1 = 1
    exact = torch.tensor([0.95 ** 6 + 0.05 ** 6, 0.95 ** 11 + 0.05 ** 11], dtype=torch.float64)
    assert torch.allclose(mcmc.autocorrelation(x1, max_lag=2), exact, atol=0.02)


def test_run_times_each_function():
    # Time is machine-dependent, so only its bookkeeping is checked: one entry per function,
    # and proposal and accept, called once per step, each take some time.
    *_, sec = run(LOWER, upper=False)
    assert set(sec) == {"init", "proposal", "accept"}
    assert sec["proposal"] > 0 and sec["accept"] > 0
