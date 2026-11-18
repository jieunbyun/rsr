"""Learning reference states from one stored sample set (samples_store, the default) or from samples drawn
anew each round (samples_store=False)."""

import pytest
import torch

from rsr import rsr

NAMES = [f"x{i}" for i in range(8)]


def _sfun(comps_st):
    """Survives if x0 and x1 work, or if any three of x2..x7 do: a coherent binary system."""
    s = (comps_st["x0"] and comps_st["x1"]) or sum(comps_st[k] for k in NAMES[2:]) >= 3
    return float(s), int(s), None


def _states_of(x):
    return {n: int(v) for n, v in zip(NAMES, x.tolist())}


def _run(tmp_path, **kwargs):
    probs = torch.tensor([[0.3, 0.7]] * len(NAMES))
    return rsr.run_ref_extraction_by_mcs(
        sfun=_sfun, probs=probs, row_names=NAMES, n_state=2, sys_upper_st=1, unk_prob_thres=0.0,
        unk_prob_opt="abs", save_every=1000, ref_update_verbose=False, output_dir=str(tmp_path), **kwargs)


def _classify(result, x):
    up, lo = torch.load(result["refs_upper_pt_path"]), torch.load(result["refs_lower_pt_path"])
    return rsr.classify_samples_with_indices(torch.nn.functional.one_hot(x.long(), 2).int(), up, lo, return_masks=True)


def test_the_store_is_the_default_and_every_stored_sample_ends_classified_as_sfun_says(tmp_path):
    torch.manual_seed(0)
    result = _run(tmp_path, n_sample=4000, sample_batch_size=1000)
    assert result["n_store"] == 4000 and result["n_store_unknown"] == 0
    assert result["metrics_log"][-1]["p_unknown"] == 0.0
    every = torch.cartesian_prod(*[torch.tensor([0, 1])] * len(NAMES))   # all 256 vectors: the rules are exact
    r = _classify(result, every)
    for i in torch.where(~r["mask_unknown"])[0].tolist():
        assert int(r["mask_upper"][i]) == _sfun(_states_of(every[i]))[1]


def test_a_given_sample_set_is_learnt_from_until_each_of_its_samples_is_classified(tmp_path):
    given = torch.tensor([[0, 0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]])
    result = _run(tmp_path, samples=given, sample_batch_size=2)
    assert result["n_store"] == 5 and result["n_store_unknown"] == 0
    # at most one round per sample, besides the round that finds none left and the final entry
    assert len(result["metrics_log"]) - 2 <= 5
    r = _classify(result, given)
    assert not r["mask_unknown"].any()
    assert r["mask_upper"].long().tolist() == [_sfun(_states_of(x))[1] for x in given]


def test_one_hot_samples_are_taken_as_their_states(tmp_path):
    given = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]])
    a = _run(tmp_path / "a", samples=given)
    b = _run(tmp_path / "b", samples=torch.nn.functional.one_hot(given, 2))
    assert a["n_store"] == b["n_store"] == 2 and a["n_store_unknown"] == b["n_store_unknown"] == 0


def test_drawing_anew_each_round_is_kept_with_samples_store_false(tmp_path):
    torch.manual_seed(0)
    result = _run(tmp_path, samples_store=False, n_sample=2000, sample_batch_size=1000, max_rounds=40)
    assert "n_store" not in result and len(result["metrics_log"]) > 1


def test_samples_without_the_store_are_refused(tmp_path):
    with pytest.raises(ValueError, match="samples_store"):
        _run(tmp_path, samples_store=False, samples=torch.zeros(3, len(NAMES), dtype=torch.long))


def test_chunk_by_chunk_holds_one_chunk_and_still_classifies_every_sample(tmp_path, monkeypatch):
    """Each chunk is learnt from before the next is taken, and no more than a chunk's unknown samples are held."""
    held, cat = [], torch.cat

    def watching(parts, *a, **kw):
        out = cat(parts, *a, **kw)
        if out.ndim == 2 and out.dtype == torch.uint8:
            held.append(len(out))
        return out

    monkeypatch.setattr(rsr.torch, "cat", watching)
    torch.manual_seed(0)
    result = _run(tmp_path, n_sample=4000, sample_batch_size=250, store_chunk_size=500)
    assert result["n_store"] == 4000 and result["n_store_unknown"] == 0
    assert held and max(held) <= 500
    whole = _run(tmp_path / "whole", n_sample=4000, sample_batch_size=250, store_chunk_size=0)
    assert whole["n_store"] == 4000 and whole["n_store_unknown"] == 0


def test_a_chunk_left_at_the_threshold_keeps_its_unknown_samples_counted(tmp_path):
    torch.manual_seed(0)
    probs = torch.tensor([[0.3, 0.7]] * len(NAMES))
    result = rsr.run_ref_extraction_by_mcs(
        sfun=_sfun, probs=probs, row_names=NAMES, n_state=2, sys_upper_st=1, unk_prob_thres=0.2,
        unk_prob_opt="abs", n_sample=3000, sample_batch_size=500, store_chunk_size=1000, save_every=1000,
        ref_update_verbose=False, output_dir=str(tmp_path))
    last = result["metrics_log"][-1]
    assert result["n_store"] == 3000
    assert abs(last["p_unknown"] - result["n_store_unknown"] / 3000) < 1e-12
    assert abs(last["p_upper"] + last["p_lower"] + last["p_unknown"] - 1) < 1e-12


def test_with_the_store_the_default_threshold_leaves_no_sample_unknown(tmp_path):
    torch.manual_seed(0)
    result = rsr.run_ref_extraction_by_mcs(
        sfun=_sfun, probs=torch.tensor([[0.3, 0.7]] * len(NAMES)), row_names=NAMES, n_state=2, sys_upper_st=1,
        n_sample=3000, sample_batch_size=1000, save_every=1000, ref_update_verbose=False, output_dir=str(tmp_path))
    assert result["n_store"] == 3000 and result["n_store_unknown"] == 0


def test_an_empty_batch_classifies_to_nothing():
    up = lo = torch.ones(1, len(NAMES), 2, dtype=torch.int32)
    empty = torch.zeros(0, len(NAMES), 2, dtype=torch.int32)
    r = rsr.classify_samples_with_indices(empty, up, lo, return_masks=True, return_violations=True)
    assert (r["upper"], r["lower"], r["unknown"]) == (0, 0, 0)
    assert r["mask_unknown"].shape == (0,) and r["idx_unknown"].numel() == 0 and r["violations_upper"].shape == (0, 1)
    assert rsr.classify_samples(empty, up, lo) == {"upper": 0, "lower": 0, "unknown": 0}


def test_an_empty_sample_set_returns_the_starting_rules(tmp_path):
    up = rsr.from_ref_dict_to_mat({"x0": (">=", 1), "x1": (">=", 1)}, NAMES, 2).unsqueeze(0).cpu()
    result = _run(tmp_path, samples=torch.zeros(0, len(NAMES), dtype=torch.long), refs_mat_upper=up)
    assert result["n_store"] == 0 and result["n_store_unknown"] == 0
    assert torch.equal(torch.load(result["refs_upper_pt_path"]).cpu(), up)
