"""Tests for TwoStageFitnessEvaluator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from liq.evolution.fitness.two_stage import TwoStageFitnessEvaluator
from liq.evolution.protocols import FitnessEvaluator
from liq.gp.types import FitnessResult

# ------------------------------------------------------------------ #
#  Mock evaluator
# ------------------------------------------------------------------ #


class _MockEvaluator:
    """Deterministic evaluator that returns pre-configured results."""

    def __init__(self, results: list[FitnessResult]) -> None:
        self._results = results
        self.call_count = 0
        self.last_programs: list[Any] | None = None

    def evaluate(
        self,
        programs: list[Any],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        self.call_count += 1
        self.last_programs = programs
        return self._results[: len(programs)]


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

_CONTEXT: dict[str, np.ndarray] = {"x": np.array([1.0, 2.0])}


def _fr(value: float, *, stage: str = "") -> FitnessResult:
    """Shorthand for creating a FitnessResult."""
    return FitnessResult(objectives=(value,), metadata={"stage": stage})


# ------------------------------------------------------------------ #
#  Construction validation
# ------------------------------------------------------------------ #


class TestConstructionValidation:
    """Verify __init__ rejects invalid parameters and accepts valid ones."""

    def test_valid_construction(self) -> None:
        stage_a = _MockEvaluator([])
        ev = TwoStageFitnessEvaluator(stage_a=stage_a, top_k=5)
        assert ev is not None

    def test_top_k_zero_raises(self) -> None:
        stage_a = _MockEvaluator([])
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            TwoStageFitnessEvaluator(stage_a=stage_a, top_k=0)

    def test_top_k_negative_raises(self) -> None:
        stage_a = _MockEvaluator([])
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            TwoStageFitnessEvaluator(stage_a=stage_a, top_k=-1)


# ------------------------------------------------------------------ #
#  Stage A only (stage_b=None -> passthrough)
# ------------------------------------------------------------------ #


class TestStageAOnly:
    """When stage_b is None, evaluate returns stage_a results unchanged."""

    def test_passthrough_returns_stage_a_results(self) -> None:
        a_results = [_fr(0.5, stage="a"), _fr(0.7, stage="a")]
        stage_a = _MockEvaluator(a_results)
        ev = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=None, top_k=1)

        programs = ["p1", "p2"]
        result = ev.evaluate(programs, _CONTEXT)

        assert result == a_results
        assert stage_a.call_count == 1

    def test_passthrough_does_not_call_stage_b(self) -> None:
        """Confirm stage_b is never invoked when it is None."""
        a_results = [_fr(0.5)]
        stage_a = _MockEvaluator(a_results)
        ev = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=None, top_k=1)

        ev.evaluate(["p1"], _CONTEXT)
        # No stage_b to check, but stage_a was called exactly once
        assert stage_a.call_count == 1


# ------------------------------------------------------------------ #
#  Stage A + B mode
# ------------------------------------------------------------------ #


class TestTwoStageMode:
    """Top K programs get Stage B results; rest keep Stage A."""

    def test_top_k_get_stage_b_results(self) -> None:
        """With 4 programs and top_k=2, top 2 by objective get stage B."""
        a_results = [
            _fr(0.3, stage="a"),  # idx 0
            _fr(0.9, stage="a"),  # idx 1  (top 1)
            _fr(0.1, stage="a"),  # idx 2
            _fr(0.7, stage="a"),  # idx 3  (top 2)
        ]
        b_results = [
            _fr(0.95, stage="b"),  # for idx 1
            _fr(0.75, stage="b"),  # for idx 3
        ]
        stage_a = _MockEvaluator(a_results)
        stage_b = _MockEvaluator(b_results)
        ev = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=stage_b, top_k=2)

        programs = ["p0", "p1", "p2", "p3"]
        result = ev.evaluate(programs, _CONTEXT)

        # idx 0 keeps stage A (0.3)
        assert result[0].objectives == (0.3,)
        assert result[0].metadata["stage"] == "a"
        # idx 1 gets stage B (0.95)
        assert result[1].objectives == (0.95,)
        assert result[1].metadata["stage"] == "b"
        # idx 2 keeps stage A (0.1)
        assert result[2].objectives == (0.1,)
        assert result[2].metadata["stage"] == "a"
        # idx 3 gets stage B (0.75)
        assert result[3].objectives == (0.75,)
        assert result[3].metadata["stage"] == "b"

    def test_stage_b_receives_only_top_k_programs(self) -> None:
        """Stage B evaluator should only receive the top K programs."""
        a_results = [
            _fr(0.2, stage="a"),  # idx 0
            _fr(0.8, stage="a"),  # idx 1  (top)
            _fr(0.5, stage="a"),  # idx 2
        ]
        b_results = [_fr(0.85, stage="b")]
        stage_a = _MockEvaluator(a_results)
        stage_b = _MockEvaluator(b_results)
        ev = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=stage_b, top_k=1)

        programs = ["p0", "p1", "p2"]
        ev.evaluate(programs, _CONTEXT)

        assert stage_b.call_count == 1
        assert stage_b.last_programs == ["p1"]


# ------------------------------------------------------------------ #
#  top_k > population -> all go through both stages
# ------------------------------------------------------------------ #


class TestTopKExceedsPopulation:
    """When top_k >= len(programs), all programs go through stage B."""

    def test_all_programs_evaluated_by_stage_b(self) -> None:
        a_results = [_fr(0.5, stage="a"), _fr(0.3, stage="a")]
        b_results = [_fr(0.55, stage="b"), _fr(0.35, stage="b")]
        stage_a = _MockEvaluator(a_results)
        stage_b = _MockEvaluator(b_results)
        ev = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=stage_b, top_k=100)

        programs = ["p0", "p1"]
        result = ev.evaluate(programs, _CONTEXT)

        # All replaced by stage B
        assert result[0].metadata["stage"] == "b"
        assert result[1].metadata["stage"] == "b"
        assert stage_b.last_programs == ["p0", "p1"]


# ------------------------------------------------------------------ #
#  top_k = 1 -> only the best goes through Stage B
# ------------------------------------------------------------------ #


class TestTopKEqualsOne:
    """When top_k=1, only the single best program gets Stage B."""

    def test_only_best_gets_stage_b(self) -> None:
        a_results = [
            _fr(0.1, stage="a"),
            _fr(0.9, stage="a"),  # best
            _fr(0.4, stage="a"),
        ]
        b_results = [_fr(0.99, stage="b")]
        stage_a = _MockEvaluator(a_results)
        stage_b = _MockEvaluator(b_results)
        ev = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=stage_b, top_k=1)

        result = ev.evaluate(["p0", "p1", "p2"], _CONTEXT)

        assert result[0].metadata["stage"] == "a"
        assert result[1].metadata["stage"] == "b"
        assert result[1].objectives == (0.99,)
        assert result[2].metadata["stage"] == "a"


# ------------------------------------------------------------------ #
#  Empty programs list -> empty result
# ------------------------------------------------------------------ #


class TestEmptyPrograms:
    """Empty program list produces empty results."""

    def test_empty_programs_returns_empty(self) -> None:
        stage_a = _MockEvaluator([])
        stage_b = _MockEvaluator([])
        ev = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=stage_b, top_k=5)

        result = ev.evaluate([], _CONTEXT)

        assert result == []


# ------------------------------------------------------------------ #
#  Single program -> goes through both stages
# ------------------------------------------------------------------ #


class TestSingleProgram:
    """A single program goes through both stages."""

    def test_single_program_gets_stage_b(self) -> None:
        a_results = [_fr(0.5, stage="a")]
        b_results = [_fr(0.8, stage="b")]
        stage_a = _MockEvaluator(a_results)
        stage_b = _MockEvaluator(b_results)
        ev = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=stage_b, top_k=1)

        result = ev.evaluate(["p0"], _CONTEXT)

        assert len(result) == 1
        assert result[0].objectives == (0.8,)
        assert result[0].metadata["stage"] == "b"


# ------------------------------------------------------------------ #
#  Original index ordering preserved
# ------------------------------------------------------------------ #


class TestIndexOrdering:
    """Results must be in the same order as input programs."""

    def test_ordering_preserved(self) -> None:
        a_results = [
            _fr(0.9, stage="a"),  # idx 0 - top 1
            _fr(0.1, stage="a"),  # idx 1
            _fr(0.8, stage="a"),  # idx 2 - top 2
            _fr(0.2, stage="a"),  # idx 3
        ]
        b_results = [
            _fr(0.95, stage="b"),  # for idx 0
            _fr(0.85, stage="b"),  # for idx 2
        ]
        stage_a = _MockEvaluator(a_results)
        stage_b = _MockEvaluator(b_results)
        ev = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=stage_b, top_k=2)

        result = ev.evaluate(["p0", "p1", "p2", "p3"], _CONTEXT)

        assert len(result) == 4
        # idx 0 (top) -> stage B
        assert result[0].objectives == (0.95,)
        # idx 1 (not top) -> stage A
        assert result[1].objectives == (0.1,)
        # idx 2 (top) -> stage B
        assert result[2].objectives == (0.85,)
        # idx 3 (not top) -> stage A
        assert result[3].objectives == (0.2,)


# ------------------------------------------------------------------ #
#  Top K selects highest fitness (descending by objectives[0])
# ------------------------------------------------------------------ #


class TestTopKSelection:
    """Top K is selected by descending objectives[0]."""

    def test_selects_highest_objectives(self) -> None:
        a_results = [
            _fr(0.3),
            _fr(0.7),  # top 2
            _fr(0.5),
            _fr(0.9),  # top 1
            _fr(0.1),
        ]
        b_results = [_fr(1.0, stage="b"), _fr(0.8, stage="b")]
        stage_a = _MockEvaluator(a_results)
        stage_b = _MockEvaluator(b_results)
        ev = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=stage_b, top_k=2)

        ev.evaluate(["p0", "p1", "p2", "p3", "p4"], _CONTEXT)

        # Stage B should receive programs at indices 3 (0.9) and 1 (0.7)
        assert stage_b.last_programs == ["p3", "p1"]


# ------------------------------------------------------------------ #
#  evaluate_fitness() delegates to evaluate()
# ------------------------------------------------------------------ #


class TestEvaluateFitnessDelegate:
    """evaluate_fitness delegates to evaluate."""

    def test_evaluate_fitness_delegates(self) -> None:
        a_results = [_fr(0.5, stage="a")]
        stage_a = _MockEvaluator(a_results)
        ev = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=None, top_k=1)

        result_eval = ev.evaluate(["p0"], _CONTEXT)
        result_compat = ev.evaluate_fitness(["p0"], _CONTEXT)

        assert result_eval[0].objectives == result_compat[0].objectives


# ------------------------------------------------------------------ #
#  Protocol compliance
# ------------------------------------------------------------------ #


class TestProtocolCompliance:
    """TwoStageFitnessEvaluator satisfies the FitnessEvaluator protocol."""

    def test_isinstance_fitness_evaluator(self) -> None:
        stage_a = _MockEvaluator([])
        ev = TwoStageFitnessEvaluator(stage_a=stage_a, top_k=1)
        assert isinstance(ev, FitnessEvaluator)
