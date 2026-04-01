"""
Tests for functions that had no coverage in test_rsr.py:
  - sample_categorical
  - classify_samples
  - classify_samples_with_indices
  - mixed_sort_key
  - sample_new_comp_st_to_test
  - get_comp_cond_sys_prob  (conditioning case)
  - get_comp_cond_sys_prob_multi (conditioning case)
"""
import pytest
import torch
from rsr import rsr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _t(*vals, dtype=torch.int32):
    """Shorthand: build a tensor on DEVICE."""
    return torch.tensor(vals, dtype=dtype, device=DEVICE)


def _fp(*vals):
    """Shorthand: build a float32 prob tensor on DEVICE."""
    return torch.tensor(vals, dtype=torch.float32, device=DEVICE)


# ===========================================================================
# sample_categorical
# ===========================================================================

class TestSampleCategorical:

    def test_output_shape(self):
        probs = _fp([0.3, 0.7], [0.5, 0.5], [0.1, 0.9])  # 3 vars, 2 states
        samples = rsr.sample_categorical(probs, 50)
        assert samples.shape == (50, 3, 2)

    def test_one_hot_per_variable(self):
        """Each (sample, variable) row must contain exactly one 1."""
        probs = _fp([0.3, 0.7], [0.5, 0.5])
        samples = rsr.sample_categorical(probs, 200)
        assert torch.all(samples.sum(dim=2) == 1), "Each variable row must be one-hot"

    def test_degenerate_probs_always_picks_forced_state(self):
        """Probability mass of 1.0 on a single state must always produce that state."""
        probs = _fp([1.0, 0.0], [0.0, 1.0])  # var0 → state 0, var1 → state 1
        samples = rsr.sample_categorical(probs, 30)
        assert torch.all(samples[:, 0, 0] == 1), "var0 must always be state 0"
        assert torch.all(samples[:, 0, 1] == 0)
        assert torch.all(samples[:, 1, 1] == 1), "var1 must always be state 1"
        assert torch.all(samples[:, 1, 0] == 0)

    def test_statistical_distribution_two_states(self):
        """Empirical proportions should match declared probabilities within tolerance."""
        torch.manual_seed(0)
        p = 0.3
        probs = _fp([1 - p, p])   # 1 variable, 2 states
        samples = rsr.sample_categorical(probs, 200_000)
        empirical = samples[:, 0, :].float().mean(dim=0)
        assert empirical[0].item() == pytest.approx(1 - p, abs=2e-2)
        assert empirical[1].item() == pytest.approx(p, abs=2e-2)

    def test_three_states_distribution(self):
        """Works correctly with more than 2 states."""
        torch.manual_seed(1)
        probs = _fp([0.2, 0.5, 0.3])  # 1 variable, 3 states
        samples = rsr.sample_categorical(probs, 200_000)
        assert samples.shape == (200_000, 1, 3)
        assert torch.all(samples.sum(dim=2) == 1)
        empirical = samples[:, 0, :].float().mean(dim=0)
        for i, expected_p in enumerate([0.2, 0.5, 0.3]):
            assert empirical[i].item() == pytest.approx(expected_p, abs=2e-2)

    def test_single_sample(self):
        """n_sample=1 still produces the right shape and a valid one-hot."""
        probs = _fp([0.5, 0.5], [0.2, 0.8])
        samples = rsr.sample_categorical(probs, 1)
        assert samples.shape == (1, 2, 2)
        assert torch.all(samples.sum(dim=2) == 1)


# ===========================================================================
# classify_samples
# ===========================================================================

class TestClassifySamples:
    """
    Rule semantics: a rule tensor (n_var, n_state) has 1 where a state is
    "allowed".  A sample is a subset of a rule iff (sample & rule) == sample
    for every variable.
    """

    # convenience: 2 vars, 2 states
    # rule_var0_state1: var0 must be in state 1; var1 unconstrained
    rule_var0_state1 = _t([0, 1], [1, 1])
    # rule_var0_state0: var0 must be in state 0; var1 unconstrained
    rule_var0_state0 = _t([1, 0], [1, 1])

    def test_all_survival(self):
        samples = _t([[0, 1], [1, 0]],
                     [[0, 1], [0, 1]])   # both have var0=state1
        counts = rsr.classify_samples(samples, [self.rule_var0_state1], [])
        assert counts == {'upper': 2, 'lower': 0, 'unknown': 0}

    def test_all_failure(self):
        samples = _t([[1, 0], [1, 0]],
                     [[1, 0], [0, 1]])   # both have var0=state0
        counts = rsr.classify_samples(samples, [], [self.rule_var0_state0])
        assert counts == {'upper': 0, 'lower': 2, 'unknown': 0}

    def test_all_unknown_when_no_rules(self):
        samples = _t([[0, 1], [1, 0]],
                     [[1, 0], [0, 1]])
        counts = rsr.classify_samples(samples, [], [])
        assert counts == {'upper': 0, 'lower': 0, 'unknown': 2}

    def test_mixed_survival_failure(self):
        samples = _t([[0, 1], [1, 0]],   # var0=1 → survival
                     [[1, 0], [0, 1]])   # var0=0 → failure
        counts = rsr.classify_samples(
            samples, [self.rule_var0_state1], [self.rule_var0_state0]
        )
        assert counts == {'upper': 1, 'lower': 1, 'unknown': 0}

    def test_mixed_with_unknown(self):
        # rule only covers var0=state1; sample with var0=state0 & no lower rule → unknown
        samples = _t([[0, 1], [1, 0]],   # survival
                     [[1, 0], [0, 1]])   # unknown (no lower rule)
        counts = rsr.classify_samples(samples, [self.rule_var0_state1], [])
        assert counts == {'upper': 1, 'lower': 0, 'unknown': 1}

    def test_counts_sum_to_n_sample(self):
        samples = _t([[0, 1], [1, 0]],
                     [[1, 0], [0, 1]],
                     [[0, 1], [0, 1]])
        counts = rsr.classify_samples(samples, [self.rule_var0_state1], [])
        total = counts['upper'] + counts['lower'] + counts['unknown']
        assert total == 3

    def test_survival_rule_checked_before_failure(self):
        """A sample matched by both rules gets classified as survival (surv comes first)."""
        both_allowed = _t([1, 1], [1, 1])   # allows all states → matches any sample
        sample = _t([[0, 1], [0, 1]])
        counts = rsr.classify_samples(sample, [both_allowed], [both_allowed])
        assert counts['upper'] == 1
        assert counts['lower'] == 0

    def test_three_state_components(self):
        """Works with 3 states per component."""
        # upper rule: var0 in state 2
        surv_rule = _t([0, 0, 1], [1, 1, 1])   # var0=state2, var1=any
        samples = _t([[0, 0, 1], [1, 0, 0]],    # var0=state2 → survival
                     [[1, 0, 0], [0, 1, 0]])    # var0=state0 → unknown
        counts = rsr.classify_samples(samples, [surv_rule], [])
        assert counts['upper'] == 1
        assert counts['unknown'] == 1


# ===========================================================================
# classify_samples_with_indices
# ===========================================================================

class TestClassifySamplesWithIndices:

    rule_s = _t([0, 1], [1, 1])   # survival: var0=state1
    rule_f = _t([1, 0], [1, 1])   # failure:  var0=state0

    def test_basic_counts_and_indices(self):
        samples = _t([[0, 1], [1, 0]],   # idx 0 → survival
                     [[1, 0], [0, 1]])   # idx 1 → failure
        result = rsr.classify_samples_with_indices(
            samples, [self.rule_s], [self.rule_f]
        )
        assert result['upper'] == 1
        assert result['lower'] == 1
        assert result['unknown'] == 0
        assert result['idx_upper'].tolist() == [0]
        assert result['idx_lower'].tolist() == [1]
        assert result['idx_unknown'].numel() == 0

    def test_all_unknown(self):
        samples = _t([[0, 1], [1, 0]],
                     [[1, 0], [0, 1]])
        result = rsr.classify_samples_with_indices(samples, [], [])
        assert result['unknown'] == 2
        assert sorted(result['idx_unknown'].tolist()) == [0, 1]
        assert result['idx_upper'].numel() == 0

    def test_return_masks_flag(self):
        samples = _t([[0, 1], [1, 0]],   # survival
                     [[1, 0], [0, 1]])   # unknown
        result = rsr.classify_samples_with_indices(
            samples, [self.rule_s], [], return_masks=True
        )
        assert 'mask_upper' in result
        assert 'mask_lower' in result
        assert 'mask_unknown' in result
        assert result['mask_upper'].tolist() == [True, False]
        assert result['mask_unknown'].tolist() == [False, True]

    def test_non_contiguous_survival_indices(self):
        """Survival samples at non-contiguous positions are all reported."""
        samples = _t([[1, 0], [1, 0]],   # idx 0 → unknown
                     [[0, 1], [0, 1]],   # idx 1 → survival
                     [[1, 0], [0, 1]],   # idx 2 → unknown
                     [[0, 1], [1, 0]])   # idx 3 → survival
        result = rsr.classify_samples_with_indices(samples, [self.rule_s], [])
        assert sorted(result['idx_upper'].tolist()) == [1, 3]
        assert sorted(result['idx_unknown'].tolist()) == [0, 2]

    def test_consistency_with_classify_samples(self):
        """Both classify functions must report identical counts."""
        samples = _t([[0, 1], [1, 0]],   # survival
                     [[1, 0], [0, 1]],   # failure
                     [[1, 0], [1, 0]])   # unknown (no fail rule matches [1,0],[1,0])
        # fail rule requires var1=state1; sample 2 has var1=state0 → not matched
        c1 = rsr.classify_samples(samples, [self.rule_s], [self.rule_f])
        c2 = rsr.classify_samples_with_indices(samples, [self.rule_s], [self.rule_f])
        assert c1['upper'] == c2['upper']
        assert c1['lower'] == c2['lower']
        assert c1['unknown'] == c2['unknown']

    def test_3d_tensor_rule_input(self):
        """Rules passed as a 3-D tensor (n_rules, n_var, n_state) are handled correctly."""
        # 1 upper rule: var0 must be state 1, var1 unconstrained → (1, 2, 2)
        rules_3d = _t([0, 1], [1, 1]).unsqueeze(0)   # (2, 2) → (1, 2, 2)
        # 1 sample: var0=state1, var1=state0 → (1, 2, 2)
        samples = _t([0, 1], [1, 0]).unsqueeze(0)
        result = rsr.classify_samples_with_indices(samples, rules_3d, [])
        assert result['upper'] == 1
        assert result['unknown'] == 0


# ===========================================================================
# mixed_sort_key
# ===========================================================================

class TestMixedSortKey:

    def test_numeric_before_string(self):
        assert rsr.mixed_sort_key(1.0) < rsr.mixed_sort_key("a")

    def test_string_before_none(self):
        assert rsr.mixed_sort_key("z") < rsr.mixed_sort_key(None)

    def test_numeric_before_none(self):
        assert rsr.mixed_sort_key(100.0) < rsr.mixed_sort_key(None)

    def test_numeric_ordering_by_value(self):
        vals = [-10.0, -1, 0, 0.5, 10]
        keys = [rsr.mixed_sort_key(v) for v in vals]
        assert keys == sorted(keys)

    def test_string_sorting_is_case_insensitive(self):
        assert rsr.mixed_sort_key("Apple") == rsr.mixed_sort_key("apple")

    def test_nan_sorts_after_finite_numerics(self):
        # NaN gets (0, 1, ...) vs finite (0, 0, v, ...) so NaN > any finite numeric
        assert rsr.mixed_sort_key(float('nan')) > rsr.mixed_sort_key(1e9)

    def test_nan_sorts_before_strings(self):
        # NaN still has first element 0 (numeric bucket), string has 1
        assert rsr.mixed_sort_key(float('nan')) < rsr.mixed_sort_key("a")

    def test_none_is_the_largest_key(self):
        candidates = [0, -1, "foo", float('nan')]
        none_key = rsr.mixed_sort_key(None)
        assert all(none_key > rsr.mixed_sort_key(v) for v in candidates)

    def test_bool_not_treated_as_numeric(self):
        # bool is excluded from the numeric path by `not isinstance(x, bool)`
        key_bool = rsr.mixed_sort_key(True)
        # bool falls through to the string bucket (first element == 1)
        assert key_bool[0] == 1

    def test_sorting_mixed_list(self):
        """sorted() on a mixed list should put numerics first, then strings, then None."""
        mixed = [None, "banana", 3, "apple", 0.5, float('nan')]
        result = sorted(mixed, key=rsr.mixed_sort_key)
        # None must be last
        assert result[-1] is None
        # Locate each category (NaN != NaN so check via isinstance + isnan)
        numeric_positions = [i for i, v in enumerate(result)
                             if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v)]
        nan_positions     = [i for i, v in enumerate(result)
                             if isinstance(v, float) and v != v]
        str_positions     = [i for i, v in enumerate(result) if isinstance(v, str)]
        assert max(numeric_positions) < min(nan_positions),  "numerics must come before NaN"
        assert max(nan_positions)     < min(str_positions),  "NaN must come before strings"
        assert max(str_positions)     < (len(result) - 1),   "strings must come before None"


# ===========================================================================
# sample_new_comp_st_to_test
# ===========================================================================

class TestSampleNewCompStToTest:

    def test_empty_rules_returns_all_ones(self):
        """With no existing rules, the function returns an all-ones tensor immediately."""
        probs = _fp([0.5, 0.5], [0.3, 0.7])
        n_comp, n_state = probs.shape
        rules_mat = torch.empty((0, n_comp, n_state), dtype=torch.int32, device=DEVICE)
        sample, _ = rsr.sample_new_comp_st_to_test(probs, rules_mat)
        assert sample.shape == (n_comp, n_state)
        assert torch.all(sample == 1)

    def test_empty_rules_all_samples_shape(self):
        """all_samples tensor has correct dimensions on the empty-rules fast path."""
        probs = _fp([0.5, 0.5], [0.3, 0.7])
        n_comp, n_state = probs.shape
        rules_mat = torch.empty((0, n_comp, n_state), dtype=torch.int32, device=DEVICE)
        _, all_samples = rsr.sample_new_comp_st_to_test(probs, rules_mat)
        assert all_samples.shape[1] == n_comp
        assert all_samples.shape[2] == n_state

    def test_with_rule_output_not_covered(self):
        """Returned sample must not already be a subset of any known rule."""
        torch.manual_seed(42)
        n_comp, n_state = 3, 2
        probs = _fp([0.5, 0.5], [0.5, 0.5], [0.5, 0.5])
        # Rule: all components must be in state 1 (only one of the two states covered).
        # Use _t(row0, row1, row2) which gives shape (3, 2), then unsqueeze to (1, 3, 2).
        rules_mat = _t([0, 1], [0, 1], [0, 1]).unsqueeze(0)   # (1, 3, 2)
        sample, _ = rsr.sample_new_comp_st_to_test(probs, rules_mat, B=128, max_iters=200)
        if sample is not None:
            assert sample.shape == (n_comp, n_state)
            is_sub, _ = rsr.is_subset(sample, rules_mat)
            assert not is_sub, "Returned candidate must not be covered by existing rules"


# ===========================================================================
# get_comp_cond_sys_prob — conditioning case
# ===========================================================================

@pytest.fixture
def five_comp_system():
    """
    Five binary components; s_fun returns 1 (survival) iff ALL components are in state 1.
    With all probs = 0.9, P(survival) = 0.9^5 ≈ 0.59049.
    """
    row_names = [f"x{i}" for i in range(1, 6)]
    probs = torch.full((5, 2), fill_value=0.0, dtype=torch.float32, device=DEVICE)
    probs[:, 0] = 0.1   # P(state=0) = 0.1
    probs[:, 1] = 0.9   # P(state=1) = 0.9

    def s_fun(comps):
        sys_st = 1 if all(v == 1 for v in comps.values()) else 0
        return None, sys_st, None

    # Empty rule matrices (let s_fun resolve everything)
    n_vars = len(row_names)
    rules_upper = torch.empty((0, n_vars, 2), dtype=torch.int32, device=DEVICE)
    rules_lower = torch.empty((0, n_vars, 2), dtype=torch.int32, device=DEVICE)
    return probs, row_names, s_fun, rules_upper, rules_lower


def test_get_comp_cond_sys_prob_unconditional(five_comp_system):
    probs, row_names, s_fun, r_surv, r_fail = five_comp_system
    torch.manual_seed(0)
    result = rsr.get_comp_cond_sys_prob(
        r_surv, r_fail, probs, {}, row_names, s_fun,
        sys_upper_st=1, n_sample=200_000, n_batch=200_000
    )
    # P(all 5 in state 1) = 0.9^5 ≈ 0.59049
    assert result['upper'] == pytest.approx(0.59049, rel=2e-2, abs=1e-2)
    assert result['upper'] + result['lower'] == pytest.approx(1.0, abs=1e-6)


def test_get_comp_cond_sys_prob_conditioned_on_failure(five_comp_system):
    """Conditioning x1=0 forces failure (s_fun needs all state 1)."""
    probs, row_names, s_fun, r_surv, r_fail = five_comp_system
    torch.manual_seed(0)
    result = rsr.get_comp_cond_sys_prob(
        r_surv, r_fail, probs, {'x1': 0}, row_names, s_fun,
        sys_upper_st=1, n_sample=10_000, n_batch=10_000
    )
    assert result['upper'] == pytest.approx(0.0, abs=1e-6)
    assert result['lower'] == pytest.approx(1.0, abs=1e-6)


def test_get_comp_cond_sys_prob_conditioned_on_survival(five_comp_system):
    """Conditioning all components to state 1 should give P(survival)=1."""
    probs, row_names, s_fun, r_surv, r_fail = five_comp_system
    cond = {name: 1 for name in row_names}
    torch.manual_seed(0)
    result = rsr.get_comp_cond_sys_prob(
        r_surv, r_fail, probs, cond, row_names, s_fun,
        sys_upper_st=1, n_sample=10_000, n_batch=10_000
    )
    assert result['upper'] == pytest.approx(1.0, abs=1e-6)


def test_get_comp_cond_sys_prob_unknown_component_raises(five_comp_system):
    probs, row_names, s_fun, r_surv, r_fail = five_comp_system
    with pytest.raises(ValueError, match="not found in row_names"):
        rsr.get_comp_cond_sys_prob(
            r_surv, r_fail, probs, {'x99': 1}, row_names, s_fun,
        )


# ===========================================================================
# get_comp_cond_sys_prob_multi — conditioning case
# ===========================================================================

def test_get_comp_cond_sys_prob_multi_conditioned_on_failure(five_comp_system):
    """Conditioning x1=0 forces all samples to fail for the AND-system."""
    probs, row_names, s_fun, _, _ = five_comp_system
    n_vars = len(row_names)
    device = DEVICE
    rules_dict_upper = {1: torch.empty((0, n_vars, 2), dtype=torch.int32, device=device)}
    rules_dict_lower = {1: torch.empty((0, n_vars, 2), dtype=torch.int32, device=device)}
    torch.manual_seed(0)
    result = rsr.get_comp_cond_sys_prob_multi(
        rules_dict_upper, rules_dict_lower, probs,
        comps_st_cond={'x1': 0},
        row_names=row_names,
        s_fun=s_fun,
        n_sample=10_000, n_batch=10_000,
    )
    assert result[0] == pytest.approx(1.0, abs=1e-6)   # state 0 (failure)
    assert result[1] == pytest.approx(0.0, abs=1e-6)   # state 1 (survival)


def test_get_comp_cond_sys_prob_multi_conditioned_all_survival(five_comp_system):
    """All components at state 1 → only survival state has probability."""
    probs, row_names, s_fun, _, _ = five_comp_system
    n_vars = len(row_names)
    device = DEVICE
    rules_dict_upper = {1: torch.empty((0, n_vars, 2), dtype=torch.int32, device=device)}
    rules_dict_lower = {1: torch.empty((0, n_vars, 2), dtype=torch.int32, device=device)}
    cond = {name: 1 for name in row_names}
    torch.manual_seed(0)
    result = rsr.get_comp_cond_sys_prob_multi(
        rules_dict_upper, rules_dict_lower, probs,
        comps_st_cond=cond,
        row_names=row_names,
        s_fun=s_fun,
        n_sample=10_000, n_batch=10_000,
    )
    assert result[1] == pytest.approx(1.0, abs=1e-6)
    assert result[0] == pytest.approx(0.0, abs=1e-6)
