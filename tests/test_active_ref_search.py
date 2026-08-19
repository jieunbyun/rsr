"""Deficit-based (active) acquisition of reference states.

Fixture: two components, each taking states 0..4, so 25 vectors in total.

    lower reference set   {(1, 2)}
    upper reference set   {(1, 4), (4, 0)}

These are consistent: the lower reference state dominates neither upper
reference state, and the two upper reference states are mutually incomparable.
Eleven of the twenty-five vectors are unresolved under them.

The candidate set is five of those eleven unresolved vectors, and the three
headline test cases below exercise the acquisition weight at 0, 1 and 2.
"""

from itertools import permutations, product

import pytest
import torch

from rsr import rsr


N_STATE = 5
ROW_NAMES = ["x1", "x2"]

# Reference sets as component-state coordinates.
REFS_UPPER = torch.tensor([[1, 4], [4, 0]], dtype=torch.int64)
REFS_LOWER = torch.tensor([[1, 2]], dtype=torch.int64)

# Five unresolved vectors used as the candidate set, in the documented order.
CANDIDATES = torch.tensor([[0, 4], [1, 3], [2, 0], [3, 1], [3, 3]], dtype=torch.int64)

# Expected deficits, checked before any scoring.
EXPECTED_D_UPPER = [1, 1, 2, 1, 1]
EXPECTED_D_LOWER = [2, 1, 1, 2, 3]


def _all_vectors():
    """All 25 component-state vectors of the fixture."""
    return torch.tensor(list(product(range(N_STATE), repeat=2)), dtype=torch.int64)


def _select(candidates, gamma, k=1):
    """Selection helper returning the chosen candidate vector(s)."""
    idx = rsr.select_refs_by_acquisition(
        candidates, REFS_UPPER, REFS_LOWER, gamma=gamma, k=k)
    return candidates[idx]


# --------------------------------------------------------------------------
# Fixture sanity: representation, consistency, and the deficit table
# --------------------------------------------------------------------------

def test_refs_mat_round_trip():
    """The toolkit's binary ref matrices map back to these coordinates."""
    up_mats = [
        rsr.from_ref_dict_to_mat({"x1": (">=", 1), "x2": (">=", 4)}, ROW_NAMES, N_STATE),
        rsr.from_ref_dict_to_mat({"x1": (">=", 4), "x2": (">=", 0)}, ROW_NAMES, N_STATE),
    ]
    low_mats = [
        rsr.from_ref_dict_to_mat({"x1": ("<=", 1), "x2": ("<=", 2)}, ROW_NAMES, N_STATE),
    ]

    up = rsr.refs_mat_to_states(torch.stack(up_mats), "upper").cpu()
    low = rsr.refs_mat_to_states(torch.stack(low_mats), "lower").cpu()

    assert torch.equal(up, REFS_UPPER), f"expected {REFS_UPPER.tolist()}, got {up.tolist()}"
    assert torch.equal(low, REFS_LOWER), f"expected {REFS_LOWER.tolist()}, got {low.tolist()}"


def test_reference_sets_are_consistent():
    """No lower reference state dominates an upper one."""
    rsr.validate_ref_consistency(REFS_UPPER, REFS_LOWER)


def test_inconsistent_reference_sets_are_rejected():
    """A lower state dominating an upper state is contradictory."""
    bad_lower = torch.tensor([[4, 4]], dtype=torch.int64)
    with pytest.raises(ValueError, match="inconsistent"):
        rsr.validate_ref_consistency(REFS_UPPER, bad_lower)


def test_eleven_vectors_are_unresolved():
    """Eleven of the twenty-five vectors are certified by neither set."""
    vecs = _all_vectors()
    d_up = rsr.compute_deficits(vecs, REFS_UPPER, "upper")
    d_low = rsr.compute_deficits(vecs, REFS_LOWER, "lower")

    unresolved = (d_up > 0) & (d_low > 0)
    assert int(unresolved.sum()) == 11, (
        f"expected 11 unresolved, got {int(unresolved.sum())}: "
        f"{vecs[unresolved].tolist()}")

    # The two deficits are never both zero under a consistent pair of sets.
    assert not bool(((d_up == 0) & (d_low == 0)).any())

    # Every candidate used below is drawn from the unresolved set.
    unresolved_set = {tuple(v) for v in vecs[unresolved].tolist()}
    for cand in CANDIDATES.tolist():
        assert tuple(cand) in unresolved_set, f"{cand} is not unresolved"


def test_deficit_table():
    """The deficits underpinning all three test cases."""
    d_up = rsr.compute_deficits(CANDIDATES, REFS_UPPER, "upper")
    d_low = rsr.compute_deficits(CANDIDATES, REFS_LOWER, "lower")

    assert d_up.tolist() == EXPECTED_D_UPPER, (
        f"upper deficits: expected {EXPECTED_D_UPPER}, got {d_up.tolist()}")
    assert d_low.tolist() == EXPECTED_D_LOWER, (
        f"lower deficits: expected {EXPECTED_D_LOWER}, got {d_low.tolist()}")

    # Deficits are non-negative integers.
    assert not torch.is_floating_point(d_up) and not torch.is_floating_point(d_low)
    assert bool((d_up >= 0).all()) and bool((d_low >= 0).all())


# --------------------------------------------------------------------------
# Test case one: pure exploration
# --------------------------------------------------------------------------

def test_case_one_pure_exploration():
    """Weight zero. Scores are 3, 2, 3, 3, 4 and (3, 3) is the unique maximum."""
    d_up = rsr.compute_deficits(CANDIDATES, REFS_UPPER, "upper")
    d_low = rsr.compute_deficits(CANDIDATES, REFS_LOWER, "lower")
    scores = rsr.acquisition_score(d_up, d_low, gamma=0.0)

    assert scores.tolist() == [3.0, 2.0, 3.0, 3.0, 4.0], scores.tolist()

    best = float(scores.max())
    assert best == 4.0
    assert int((scores == best).sum()) == 1, "the maximum should be unique here"

    chosen = _select(CANDIDATES, gamma=0.0)
    assert chosen.tolist() == [[3, 3]], chosen.tolist()


# --------------------------------------------------------------------------
# Test case two: the switching boundary
# --------------------------------------------------------------------------

def test_case_two_switching_boundary_is_a_five_way_tie():
    """Weight one. All five candidates score two."""
    d_up = rsr.compute_deficits(CANDIDATES, REFS_UPPER, "upper")
    d_low = rsr.compute_deficits(CANDIDATES, REFS_LOWER, "lower")
    scores = rsr.acquisition_score(d_up, d_low, gamma=1.0)

    assert scores.tolist() == [2.0, 2.0, 2.0, 2.0, 2.0], scores.tolist()
    assert int((scores == 2.0).sum()) == 5, "expected an exact five-way tie"


def test_case_two_tie_is_broken_lexicographically():
    """The tie resolves to (0, 4), the lexicographically smallest candidate."""
    chosen = _select(CANDIDATES, gamma=1.0)
    assert chosen.tolist() == [[0, 4]], chosen.tolist()


def test_case_two_selection_is_independent_of_input_order():
    """All 120 orderings of the tied candidates give the same selection.

    This is the point of the case: the tie must not be resolved by the order
    in which candidates happen to be supplied, nor by floating-point noise.
    """
    rows = CANDIDATES.tolist()
    for perm in permutations(range(len(rows))):
        shuffled = torch.tensor([rows[i] for i in perm], dtype=torch.int64)
        chosen = _select(shuffled, gamma=1.0)
        assert chosen.tolist() == [[0, 4]], (
            f"order {perm} gave {chosen.tolist()}, expected [[0, 4]]")


def test_case_two_full_ranking_is_lexicographic():
    """With every score tied, the whole ranking is the lexicographic order."""
    chosen = _select(CANDIDATES, gamma=1.0, k=5)
    assert chosen.tolist() == [[0, 4], [1, 3], [2, 0], [3, 1], [3, 3]], chosen.tolist()


# --------------------------------------------------------------------------
# Test case three: uncertainty-dominated
# --------------------------------------------------------------------------

def test_case_three_uncertainty_dominated():
    """Weight two. Scores are 1, 2, 1, 1, 0 and (1, 3) is the unique maximum."""
    d_up = rsr.compute_deficits(CANDIDATES, REFS_UPPER, "upper")
    d_low = rsr.compute_deficits(CANDIDATES, REFS_LOWER, "lower")
    scores = rsr.acquisition_score(d_up, d_low, gamma=2.0)

    assert scores.tolist() == [1.0, 2.0, 1.0, 1.0, 0.0], scores.tolist()

    best = float(scores.max())
    assert best == 2.0
    assert int((scores == best).sum()) == 1, "the maximum should be unique here"

    chosen = _select(CANDIDATES, gamma=2.0)
    assert chosen.tolist() == [[1, 3]], chosen.tolist()

    # The balanced candidate wins because its two deficits are equal, ...
    idx = 1
    assert int(d_up[idx]) == int(d_low[idx])

    # ... while (3, 3), which won test case one, now scores lowest of the five.
    assert float(scores[4]) == float(scores.min())


# --------------------------------------------------------------------------
# Further checks
# --------------------------------------------------------------------------

def test_certified_vectors_are_never_selected():
    """(0, 0) is certified failing and (4, 4) certified surviving."""
    certified = torch.tensor([[0, 0], [4, 4]], dtype=torch.int64)
    d_up = rsr.compute_deficits(certified, REFS_UPPER, "upper")
    d_low = rsr.compute_deficits(certified, REFS_LOWER, "lower")

    assert int(d_low[0]) == 0, "(0, 0) should be certified failing"
    assert int(d_up[1]) == 0, "(4, 4) should be certified surviving"

    # Mixed in with a genuine candidate, neither may be chosen at any weight.
    mixed = torch.cat([certified, CANDIDATES])
    for gamma in (0.0, 1.0, 2.0):
        chosen = _select(mixed, gamma=gamma, k=len(mixed))
        picked = {tuple(v) for v in chosen.tolist()}
        assert (0, 0) not in picked and (4, 4) not in picked, picked
        assert len(picked) == len(CANDIDATES), "only the unresolved ones rank"


def test_reference_states_are_themselves_certified():
    """Each reference state is certified by its own set."""
    d_low_of_lower = rsr.compute_deficits(REFS_LOWER, REFS_LOWER, "lower")
    assert d_low_of_lower.tolist() == [0], d_low_of_lower.tolist()

    d_up_of_upper = rsr.compute_deficits(REFS_UPPER, REFS_UPPER, "upper")
    assert d_up_of_upper.tolist() == [0, 0], d_up_of_upper.tolist()


def test_deficits_respect_the_partial_order():
    """Along an increasing chain, d+ is non-increasing and d- non-decreasing."""
    chain = torch.tensor([[0, 0], [0, 3], [1, 3], [3, 3]], dtype=torch.int64)

    # The chain really is increasing.
    for a, b in zip(chain[:-1], chain[1:]):
        assert bool((a <= b).all()) and bool((a < b).any())

    d_up = rsr.compute_deficits(chain, REFS_UPPER, "upper").tolist()
    d_low = rsr.compute_deficits(chain, REFS_LOWER, "lower").tolist()

    assert d_up == sorted(d_up, reverse=True), f"d+ not non-increasing: {d_up}"
    assert d_low == sorted(d_low), f"d- not non-decreasing: {d_low}"


def test_no_unresolved_candidate_raises():
    """A fully certified candidate set is an error, not an arbitrary choice."""
    certified = torch.tensor([[0, 0], [4, 4]], dtype=torch.int64)
    with pytest.raises(ValueError, match="no unresolved candidate"):
        rsr.select_refs_by_acquisition(certified, REFS_UPPER, REFS_LOWER, gamma=1.0)


def test_empty_reference_set_raises():
    """Deficits are undefined without references to measure against."""
    empty = torch.zeros((0, 2), dtype=torch.int64)
    with pytest.raises(ValueError, match="empty"):
        rsr.compute_deficits(CANDIDATES, empty, "upper")


def test_negative_gamma_rejected():
    d_up = rsr.compute_deficits(CANDIDATES, REFS_UPPER, "upper")
    d_low = rsr.compute_deficits(CANDIDATES, REFS_LOWER, "lower")
    with pytest.raises(ValueError, match="non-negative"):
        rsr.acquisition_score(d_up, d_low, gamma=-0.5)


def test_deficits_are_chunk_invariant():
    """Chunking the reference set must not change the answer."""
    ref = rsr.compute_deficits(CANDIDATES, REFS_UPPER, "upper")
    tiny = rsr.compute_deficits(CANDIDATES, REFS_UPPER, "upper", max_elems=1)
    assert ref.tolist() == tiny.tolist()


# --------------------------------------------------------------------------
# Wiring into the reference-extraction driver
# --------------------------------------------------------------------------

def _toy_system():
    """A six-component binary system that survives once three components work."""
    names = [f"x{i}" for i in range(6)]

    def sfun(comps_st):
        total = sum(comps_st[k] for k in names)
        return total, (1 if total >= 3 else 0), None

    probs = torch.tensor([[0.5, 0.5]] * len(names))
    return sfun, probs, names


def _run_extraction(tmp_path, monkeypatch, **kwargs):
    """Run a short extraction, counting acquisition-based selections."""
    sfun, probs, names = _toy_system()

    calls = {"n": 0}
    original = rsr.select_refs_by_acquisition

    def counting(*args, **kw):
        calls["n"] += 1
        return original(*args, **kw)

    monkeypatch.setattr(rsr, "select_refs_by_acquisition", counting)

    result = rsr.run_ref_extraction_by_mcs(
        sfun=sfun, probs=probs, row_names=names, n_state=2, sys_upper_st=1,
        max_rounds=6, n_sample=2000, sample_batch_size=1000,
        prob_update_every=1000, save_every=1000, min_ref_search=False,
        n_workers=1, ref_update_verbose=False, output_dir=str(tmp_path),
        **kwargs)
    return result, calls["n"]


def test_driver_uses_acquisition_when_enabled(tmp_path, monkeypatch):
    """The default path scores unknowns instead of drawing them at random."""
    torch.manual_seed(0)
    result, n_calls = _run_extraction(tmp_path, monkeypatch,
                                      active_ref_search=True, acq_gamma=1.0)

    assert n_calls > 0, "acquisition-based selection was never used"
    assert len(result["metrics_log"]) > 0
    # Both reference sets grew, so the run really did reach the active path.
    last = result["metrics_log"][-1]
    assert last["n_refs_upper"] > 0 and last["n_refs_lower"] > 0


def test_driver_falls_back_to_random_when_disabled(tmp_path, monkeypatch):
    """Opting out restores the original uniform-random pick."""
    torch.manual_seed(0)
    result, n_calls = _run_extraction(tmp_path, monkeypatch,
                                      active_ref_search=False)

    assert n_calls == 0, "acquisition should not run when active_ref_search is off"
    assert len(result["metrics_log"]) > 0


def test_driver_rejects_negative_gamma(tmp_path, monkeypatch):
    """A negative weight is caught rather than silently inverting the score."""
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="non-negative"):
        _run_extraction(tmp_path, monkeypatch,
                        active_ref_search=True, acq_gamma=-1.0)
