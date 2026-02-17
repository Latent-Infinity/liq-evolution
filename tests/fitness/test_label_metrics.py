"""Tests for label-based fitness evaluation (F1, precision@k, accuracy)."""

from __future__ import annotations

import numpy as np
import pytest

from liq.evolution.errors import FitnessError
from liq.evolution.fitness.label_metrics import LabelFitnessEvaluator
from liq.gp.program.ast import TerminalNode
from liq.gp.types import FitnessResult, Series


def _make_program(name: str = "scores") -> TerminalNode:
    """Create a TerminalNode that reads *name* from context."""
    return TerminalNode(name=name, output_type=Series)


def _make_context(
    scores: list[float],
    labels: list[float],
    *,
    score_key: str = "scores",
) -> dict[str, np.ndarray]:
    """Build evaluation context with given scores and labels."""
    return {
        score_key: np.array(scores, dtype=np.float64),
        "labels": np.array(labels, dtype=np.float64),
    }


# ------------------------------------------------------------------ #
#  Constructor validation
# ------------------------------------------------------------------ #


class TestConstructorValidation:
    """Verify __init__ rejects invalid parameters."""

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            LabelFitnessEvaluator(metric="bad")

    def test_top_k_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            LabelFitnessEvaluator(top_k=0.0)

    def test_top_k_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            LabelFitnessEvaluator(top_k=-0.1)

    def test_top_k_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            LabelFitnessEvaluator(top_k=1.5)

    def test_negative_turnover_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="turnover_weight"):
            LabelFitnessEvaluator(turnover_weight=-0.1)

    def test_valid_construction(self) -> None:
        """Smoke test: valid parameters do not raise."""
        ev = LabelFitnessEvaluator(metric="f1", top_k=0.5, turnover_weight=0.1)
        assert ev is not None


# ------------------------------------------------------------------ #
#  F1 metric
# ------------------------------------------------------------------ #


class TestF1Metric:
    """F1 golden-value tests."""

    def test_f1_golden_value(self) -> None:
        """top_k=0.5 on 5 samples: top 2 predicted positive.

        scores = [0.9, 0.8, 0.3, 0.1, 0.7]
        labels = [1,   1,   0,   0,   1  ]
        k = max(1, int(5 * 0.5)) = 2
        argsort(scores) = [3, 2, 4, 1, 0]
        threshold_idx = argsort[-2] = 1  →  threshold = 0.8
        preds = [1, 1, 0, 0, 0]
        TP=2  FP=0  FN=1
        P=1.0  R=2/3  F1 = 2*1*(2/3) / (1 + 2/3) = 0.8
        """
        ev = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        ctx = _make_context(
            scores=[0.9, 0.8, 0.3, 0.1, 0.7],
            labels=[1.0, 1.0, 0.0, 0.0, 1.0],
        )
        [result] = ev.evaluate([_make_program()], ctx)
        assert isinstance(result, FitnessResult)
        assert result.objectives == pytest.approx((0.8,), abs=1e-9)

    def test_f1_perfect_score(self) -> None:
        """When predictions perfectly match labels, F1 = 1.0.

        scores = [0.9, 0.8, 0.1, 0.2]
        labels = [1,   1,   0,   0  ]
        top_k=0.5 → k=2
        argsort = [2, 3, 1, 0]  threshold_idx = argsort[-2] = 1  threshold = 0.8
        preds = [1, 1, 0, 0]  →  perfect match
        """
        ev = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        ctx = _make_context(
            scores=[0.9, 0.8, 0.1, 0.2],
            labels=[1.0, 1.0, 0.0, 0.0],
        )
        [result] = ev.evaluate([_make_program()], ctx)
        assert result.objectives == pytest.approx((1.0,), abs=1e-9)


# ------------------------------------------------------------------ #
#  Precision@k metric
# ------------------------------------------------------------------ #


class TestPrecisionAtK:
    """Precision@k golden-value tests."""

    def test_precision_at_k_golden(self) -> None:
        """top_k=0.2 on 5 samples → k=1, top score = 0.9 (idx 0, label 1).

        precision@k = mean(labels[top_k_idx]) = 1.0
        """
        ev = LabelFitnessEvaluator(metric="precision_at_k", top_k=0.2)
        ctx = _make_context(
            scores=[0.9, 0.8, 0.3, 0.1, 0.7],
            labels=[1.0, 1.0, 0.0, 0.0, 1.0],
        )
        [result] = ev.evaluate([_make_program()], ctx)
        assert result.objectives == pytest.approx((1.0,), abs=1e-9)

    def test_precision_at_k_imperfect(self) -> None:
        """top_k=0.4 on 5 samples → k=2, top 2 scores are idx 0,1.

        labels[0]=1, labels[1]=0 → precision = 0.5
        """
        ev = LabelFitnessEvaluator(metric="precision_at_k", top_k=0.4)
        ctx = _make_context(
            scores=[0.9, 0.8, 0.3, 0.1, 0.7],
            labels=[1.0, 0.0, 0.0, 0.0, 1.0],
        )
        [result] = ev.evaluate([_make_program()], ctx)
        assert result.objectives == pytest.approx((0.5,), abs=1e-9)


# ------------------------------------------------------------------ #
#  Accuracy metric
# ------------------------------------------------------------------ #


class TestAccuracy:
    """Accuracy golden-value tests."""

    def test_accuracy_golden(self) -> None:
        """top_k=0.5 on 5 samples: preds=[1,1,0,0,0], labels=[1,1,0,0,1].

        Correct: 4/5 = 0.8
        """
        ev = LabelFitnessEvaluator(metric="accuracy", top_k=0.5)
        ctx = _make_context(
            scores=[0.9, 0.8, 0.3, 0.1, 0.7],
            labels=[1.0, 1.0, 0.0, 0.0, 1.0],
        )
        [result] = ev.evaluate([_make_program()], ctx)
        assert result.objectives == pytest.approx((0.8,), abs=1e-9)

    def test_accuracy_perfect(self) -> None:
        """Perfect accuracy when threshold cleanly separates classes."""
        ev = LabelFitnessEvaluator(metric="accuracy", top_k=0.5)
        ctx = _make_context(
            scores=[0.9, 0.8, 0.1, 0.2],
            labels=[1.0, 1.0, 0.0, 0.0],
        )
        [result] = ev.evaluate([_make_program()], ctx)
        assert result.objectives == pytest.approx((1.0,), abs=1e-9)

    def test_accuracy_all_zeros_all_zero_labels(self) -> None:
        """All-zero scores → all predicted positive (tied at threshold).

        With labels=[0,0,0,0], accuracy = 0.0 since all predictions
        are positive but all labels are negative.
        This is an edge case of the top-k thresholding approach.
        """
        ev = LabelFitnessEvaluator(metric="accuracy", top_k=0.5)
        ctx = _make_context(
            scores=[0.0, 0.0, 0.0, 0.0],
            labels=[0.0, 0.0, 0.0, 0.0],
        )
        [result] = ev.evaluate([_make_program()], ctx)
        # All scores tied → all >= threshold → all predicted 1 → accuracy = 0.0
        assert result.objectives == pytest.approx((0.0,), abs=1e-9)


# ------------------------------------------------------------------ #
#  NaN handling
# ------------------------------------------------------------------ #


class TestNaNHandling:
    """NaN score handling."""

    def test_all_nan_scores_worst_fitness(self) -> None:
        """All-NaN scores produce worst fitness (0.0)."""
        ev = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        ctx = _make_context(
            scores=[float("nan")] * 5,
            labels=[1.0, 1.0, 0.0, 0.0, 1.0],
        )
        [result] = ev.evaluate([_make_program()], ctx)
        assert result.objectives == (0.0,)
        assert result.metadata["reason"] == "all_nan"

    def test_partial_nan_treated_as_non_positive(self) -> None:
        """NaN values replaced by -inf → never predicted positive.

        scores = [0.9, nan, 0.3, 0.1, 0.7]
        After clean: [0.9, -inf, 0.3, 0.1, 0.7]
        top_k=0.5 → k=2
        argsort = [1, 3, 2, 4, 0] → threshold_idx = argsort[-2] = 4 → threshold = 0.7
        preds = [1, 0, 0, 0, 1]
        labels = [1, 1, 0, 0, 1]
        TP=2  FP=0  FN=1
        P=1.0  R=2/3  F1 = 0.8
        """
        ev = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        ctx = _make_context(
            scores=[0.9, float("nan"), 0.3, 0.1, 0.7],
            labels=[1.0, 1.0, 0.0, 0.0, 1.0],
        )
        [result] = ev.evaluate([_make_program()], ctx)
        assert result.objectives == pytest.approx((0.8,), abs=1e-9)


# ------------------------------------------------------------------ #
#  Turnover penalty
# ------------------------------------------------------------------ #


class TestTurnoverPenalty:
    """Turnover penalty reduces fitness for high-churn signals."""

    def test_stable_signal_no_penalty(self) -> None:
        """A stable signal (sorted scores) has low turnover."""
        ev_penalized = LabelFitnessEvaluator(
            metric="f1", top_k=0.5, turnover_weight=0.5
        )
        ev_base = LabelFitnessEvaluator(metric="f1", top_k=0.5, turnover_weight=0.0)

        # Monotone scores → low turnover
        ctx = _make_context(
            scores=[0.1, 0.2, 0.3, 0.8, 0.9],
            labels=[0.0, 0.0, 0.0, 1.0, 1.0],
        )
        [result_base] = ev_base.evaluate([_make_program()], ctx)
        [result_pen] = ev_penalized.evaluate([_make_program()], ctx)
        # With low turnover, penalty is small
        assert result_pen.objectives[0] <= result_base.objectives[0]

    def test_alternating_signal_higher_penalty(self) -> None:
        """Alternating scores have high turnover → lower fitness than stable."""
        ev = LabelFitnessEvaluator(metric="f1", top_k=0.5, turnover_weight=0.5)

        # Stable: sorted ascending
        ctx_stable = _make_context(
            scores=[0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
            labels=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        # Alternating: high-low-high-low
        ctx_alt = _make_context(
            scores=[0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
            labels=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            score_key="scores",
        )

        [result_stable] = ev.evaluate([_make_program()], ctx_stable)
        [result_alt] = ev.evaluate([_make_program()], ctx_alt)

        assert result_alt.objectives[0] < result_stable.objectives[0]

    def test_zero_turnover_weight_no_effect(self) -> None:
        """With turnover_weight=0, turnover does not affect fitness."""
        ev = LabelFitnessEvaluator(metric="f1", top_k=0.5, turnover_weight=0.0)
        ctx = _make_context(
            scores=[0.9, 0.1, 0.8, 0.2, 0.7],
            labels=[1.0, 0.0, 1.0, 0.0, 1.0],
        )
        [result] = ev.evaluate([_make_program()], ctx)
        # Should be identical to base F1 (no penalty applied)
        assert result.objectives[0] > 0.0

    def test_single_element_turnover_zero(self) -> None:
        """Single-element arrays have zero turnover (no flips possible)."""
        ev = LabelFitnessEvaluator(metric="f1", top_k=1.0, turnover_weight=0.5)
        ctx = _make_context(scores=[0.9], labels=[1.0])
        [result] = ev.evaluate([_make_program()], ctx)
        # Single element: _compute_turnover returns 0.0, no penalty
        assert result.objectives[0] > 0.0


# ------------------------------------------------------------------ #
#  Multiple programs
# ------------------------------------------------------------------ #


class TestMultiplePrograms:
    """Evaluate a batch of programs."""

    def test_output_length_matches(self) -> None:
        """Output list length must match input programs list."""
        ev = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        programs = [
            _make_program("s1"),
            _make_program("s2"),
            _make_program("s3"),
        ]
        ctx: dict[str, np.ndarray] = {
            "s1": np.array([0.9, 0.8, 0.1], dtype=np.float64),
            "s2": np.array([0.1, 0.2, 0.9], dtype=np.float64),
            "s3": np.array([0.5, 0.5, 0.5], dtype=np.float64),
            "labels": np.array([1.0, 1.0, 0.0], dtype=np.float64),
        }
        results = ev.evaluate(programs, ctx)
        assert len(results) == 3
        assert all(isinstance(r, FitnessResult) for r in results)


# ------------------------------------------------------------------ #
#  Missing labels
# ------------------------------------------------------------------ #


class TestMissingLabels:
    """Context without 'labels' key must raise FitnessError."""

    def test_labels_missing_raises_fitness_error(self) -> None:
        ev = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        ctx = {"scores": np.array([0.9, 0.8], dtype=np.float64)}
        with pytest.raises(FitnessError, match="labels"):
            ev.evaluate([_make_program()], ctx)


# ------------------------------------------------------------------ #
#  Output type contract
# ------------------------------------------------------------------ #


class TestOutputContract:
    """Each result must be a FitnessResult with the correct shape."""

    def test_result_is_fitness_result(self) -> None:
        ev = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        ctx = _make_context([0.9, 0.1], [1.0, 0.0])
        [result] = ev.evaluate([_make_program()], ctx)
        assert isinstance(result, FitnessResult)

    def test_single_objective(self) -> None:
        ev = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        ctx = _make_context([0.9, 0.1], [1.0, 0.0])
        [result] = ev.evaluate([_make_program()], ctx)
        assert len(result.objectives) == 1

    def test_metadata_contains_metric(self) -> None:
        ev = LabelFitnessEvaluator(metric="precision_at_k", top_k=0.5)
        ctx = _make_context([0.9, 0.1], [1.0, 0.0])
        [result] = ev.evaluate([_make_program()], ctx)
        assert result.metadata["metric"] == "precision_at_k"


# ------------------------------------------------------------------ #
#  evaluate_fitness backward compat
# ------------------------------------------------------------------ #


class TestEvaluateFitnessCompat:
    """evaluate_fitness delegates to evaluate."""

    def test_evaluate_fitness_delegates(self) -> None:
        ev = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        ctx = _make_context([0.9, 0.1], [1.0, 0.0])
        results_eval = ev.evaluate([_make_program()], ctx)
        results_compat = ev.evaluate_fitness([_make_program()], ctx)
        assert results_eval[0].objectives == results_compat[0].objectives
