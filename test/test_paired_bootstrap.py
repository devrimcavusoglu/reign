"""Regression tests for ``scripts/paired_bootstrap.py`` (paper Table 5).

The script decides, from per-query retrieval scores, whether REIGN differs
significantly from a baseline. Two things must hold for those verdicts to be
worth printing:

* **They must agree with a standard test.** The paired bootstrap CI and the
  sign-flip randomization test are both non-parametric, but on well-behaved
  synthetic data they should reach the same conclusion as a paired t-test
  (``scipy.stats.ttest_rel``). Tests below check agreement on a clearly
  significant case, a clearly null case, and a battery of random cases where
  the t-test is unambiguous. Borderline cases are deliberately excluded: the
  two families genuinely differ near alpha, and asserting agreement there would
  test noise.
* **They must be deterministic.** The paper quotes exact p-values, so a fixed
  ``--seed`` has to give byte-identical numbers on every run.

Fast, offline, CPU-only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

_SPEC = importlib.util.spec_from_file_location(
    "paired_bootstrap",
    Path(__file__).resolve().parents[1] / "scripts" / "paired_bootstrap.py",
)
pb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pb)

ALPHA = 0.05
N_QUERIES = 187  # the DAPFAM test split size the paper reports


def _run(sys_scores, base_scores, *, seed=42, n_boot=10000, n_perm=10000):
    return pb.paired_test(
        np.asarray(sys_scores, dtype=np.float64),
        np.asarray(base_scores, dtype=np.float64),
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
        alpha=ALPHA,
    )


def _synthetic(effect, *, seed, n=N_QUERIES, spread=0.10, noise=0.02):
    """Per-query nDCG-like pairs whose true mean difference is ``effect``.

    ``spread`` is between-query difficulty (shared by both systems, which is
    exactly what pairing removes); ``noise`` is the per-query disagreement that
    actually drives the test.
    """
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(0.30, spread, size=n)
    base = np.clip(difficulty + rng.normal(0, noise, size=n), 0.0, 1.0)
    sys_ = np.clip(difficulty + effect + rng.normal(0, noise, size=n), 0.0, 1.0)
    return sys_, base


# --------------------------------------------------------------------------
# Agreement with the paired t-test
# --------------------------------------------------------------------------


def test_clearly_significant_case_agrees_with_ttest():
    """A large, consistent improvement: every test must call it significant."""
    sys_, base = _synthetic(effect=0.05, seed=0)
    t_p = stats.ttest_rel(sys_, base).pvalue
    assert t_p < 1e-3, "fixture is not clearly significant under the t-test"

    res = _run(sys_, base)
    assert res["p_randomization"] < ALPHA
    assert res["significant"] is True
    assert res["ci_excludes_zero"] is True
    assert res["ci_lower"] > 0  # a real improvement, not just "different"
    assert res["mean_difference"] == pytest.approx(float((sys_ - base).mean()))
    # The bootstrap CI should bracket the observed difference.
    assert res["ci_lower"] < res["mean_difference"] < res["ci_upper"]


def test_clearly_null_case_agrees_with_ttest():
    """No true effect: neither the t-test nor the randomization test may fire.

    The seed is pinned to a draw the t-test calls unambiguously null (p ~ .95).
    Under a true null some draws land below alpha by chance, which is a property
    of the null, not a bug — the sweep test below covers those by only asserting
    where the t-test is decisive.
    """
    sys_, base = _synthetic(effect=0.0, seed=6)
    t_p = stats.ttest_rel(sys_, base).pvalue
    assert t_p > 0.20, "fixture is not clearly null under the t-test"

    res = _run(sys_, base)
    assert res["p_randomization"] > ALPHA
    assert res["significant"] is False
    assert res["ci_excludes_zero"] is False
    assert res["ci_lower"] < 0 < res["ci_upper"]


def test_verdicts_match_the_ttest_wherever_the_ttest_is_unambiguous():
    """Sweep effect sizes and seeds; compare verdicts where the t-test is clear.

    Cases whose t-test p-value sits near alpha are skipped rather than asserted:
    a parametric and a non-parametric test are not required to agree on a
    coin-flip, and pinning that down would make the test flaky rather than
    informative.
    """
    checked = 0
    for effect in (-0.05, -0.02, 0.0, 0.002, 0.02, 0.05):
        for seed in range(4):
            sys_, base = _synthetic(effect=effect, seed=100 + seed)
            t_p = stats.ttest_rel(sys_, base).pvalue
            res = _run(sys_, base)
            p_perm = res["p_randomization"]
            if t_p < 0.001:
                assert p_perm < ALPHA, f"effect={effect} seed={seed}: t={t_p} perm={p_perm}"
                assert res["ci_excludes_zero"] is True
                checked += 1
            elif t_p > 0.30:
                assert p_perm > ALPHA, f"effect={effect} seed={seed}: t={t_p} perm={p_perm}"
                assert res["ci_excludes_zero"] is False
                checked += 1
    assert checked >= 8, "sweep produced too few unambiguous cases to be meaningful"


def test_direction_of_the_difference_is_reported_correctly():
    """A system that is worse must produce a negative delta and a CI below zero."""
    sys_, base = _synthetic(effect=-0.05, seed=7)
    res = _run(sys_, base)
    assert res["mean_difference"] < 0
    assert res["ci_upper"] < 0
    assert res["significant"] is True
    assert res["losses"] > res["wins"]


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_fixed_seed_is_bit_reproducible():
    sys_, base = _synthetic(effect=0.01, seed=3)
    a = _run(sys_, base, seed=42)
    b = _run(sys_, base, seed=42)
    assert a == b, "same seed must reproduce every field exactly"


def test_seed_changes_only_the_resampled_quantities():
    sys_, base = _synthetic(effect=0.01, seed=3)
    a = _run(sys_, base, seed=42)
    b = _run(sys_, base, seed=7)
    # Observed statistics are computed from the data, not from the RNG.
    for key in ("n_queries", "mean_system", "mean_baseline", "mean_difference",
                "wins", "losses", "ties", "win_rate"):
        assert a[key] == b[key], f"{key} must not depend on the seed"
    # ... while the resampled CI does depend on it, and is recorded.
    assert a["seed"] == 42 and b["seed"] == 7


def test_win_loss_tie_counts_are_consistent():
    sys_, base = _synthetic(effect=0.02, seed=11)
    res = _run(sys_, base)
    assert res["wins"] + res["losses"] + res["ties"] == N_QUERIES
    assert res["win_rate"] == pytest.approx(res["wins"] / N_QUERIES)
    assert res["n_queries"] == N_QUERIES


# --------------------------------------------------------------------------
# Holm-Bonferroni correction
# --------------------------------------------------------------------------


def test_holm_matches_the_hand_computed_step_down():
    # Sorted p: .001 .010 .030 .040 with m=4.
    #   rank 0: 4 * .001 = .004
    #   rank 1: 3 * .010 = .030
    #   rank 2: 2 * .030 = .060
    #   rank 3: 1 * .040 = .060  -> raised to the running max .060 (monotone)
    raw = [0.030, 0.001, 0.040, 0.010]
    adj = pb.holm(raw)
    assert adj == pytest.approx([0.060, 0.004, 0.060, 0.030])


def test_holm_is_monotone_capped_and_never_lenient():
    rng = np.random.default_rng(0)
    for _ in range(20):
        raw = list(rng.uniform(0, 1, size=6))
        adj = pb.holm(raw)
        assert all(0.0 <= a <= 1.0 for a in adj)
        # Adjusting for multiplicity can only make a p-value larger.
        assert all(a >= r - 1e-12 for a, r in zip(adj, raw))
        # Monotone in the ranking of the raw p-values.
        order = np.argsort(raw)
        ranked = [adj[i] for i in order]
        assert all(x <= y + 1e-12 for x, y in zip(ranked, ranked[1:]))


def test_holm_is_a_no_op_for_a_single_comparison():
    assert pb.holm([0.037]) == pytest.approx([0.037])


def test_holm_across_a_family_can_flip_a_borderline_verdict():
    """The reason all five baselines go in one invocation.

    A raw p just under alpha survives on its own but not inside a family of
    five; running the comparisons separately would under-correct.
    """
    alone = pb.holm([0.045])[0]
    in_family = pb.holm([0.045, 0.2, 0.3, 0.4, 0.5])[0]
    assert alone < ALPHA <= in_family


# --------------------------------------------------------------------------
# Result-file loading and its self-check
# --------------------------------------------------------------------------


def _write_result(path, scores, metric="nDCG@100", name="sys", aggregate="mean"):
    payload = {
        "encoder": name,
        "per_query": {q: {metric: float(v)} for q, v in scores.items()},
    }
    if aggregate == "mean":
        payload["metrics"] = {metric: float(np.mean(list(scores.values())))}
    elif aggregate is not None:
        payload["metrics"] = {metric: float(aggregate)}
    Path(path).write_text(json.dumps(payload))


def test_load_per_query_reads_scores_and_name(tmp_path):
    scores = {f"Q{i}": 0.1 * i for i in range(5)}
    p = tmp_path / "r.json"
    _write_result(p, scores, name="reign-gte-large")
    name, loaded = pb.load_per_query(str(p), "nDCG@100")
    assert name == "reign-gte-large"
    assert loaded == pytest.approx(scores)


def test_load_per_query_rejects_an_aggregate_that_is_not_the_per_query_mean(tmp_path):
    """Guards against pairing a per-query dump with someone else's headline."""
    scores = {f"Q{i}": 0.1 * i for i in range(5)}
    p = tmp_path / "r.json"
    _write_result(p, scores, aggregate=0.99)
    with pytest.raises(SystemExit, match="per-query mean"):
        pb.load_per_query(str(p), "nDCG@100")


def test_load_per_query_rejects_a_file_without_per_query_scores(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"encoder": "x", "metrics": {"nDCG@100": 0.3}}))
    with pytest.raises(SystemExit, match="per_query"):
        pb.load_per_query(str(p), "nDCG@100")


def test_load_per_query_rejects_a_missing_metric(tmp_path):
    scores = {f"Q{i}": 0.1 * i for i in range(5)}
    p = tmp_path / "r.json"
    _write_result(p, scores, metric="nDCG@10")
    with pytest.raises(SystemExit, match="nDCG@100"):
        pb.load_per_query(str(p), "nDCG@100")
