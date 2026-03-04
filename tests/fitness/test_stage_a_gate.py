"""Stage-2 contract tests for Stage-A scoring and Stage-A->Stage-B gating."""

from __future__ import annotations

from typing import Any

import numpy as np

from liq.evolution.config import EvolutionRunConfig, FitnessStageConfig
from liq.evolution.fitness.label_metrics import LabelFitnessEvaluator
from liq.evolution.fitness.objectives import wire_objectives
from liq.evolution.fitness.two_stage import TwoStageFitnessEvaluator
from liq.gp.program.ast import TerminalNode
from liq.gp.types import FitnessResult, Series


def _program(name: str = "scores") -> TerminalNode:
    return TerminalNode(name=name, output_type=Series)


def _label_context(scores: list[float], labels: list[float]) -> dict[str, np.ndarray]:
    return {
        "scores": np.asarray(scores, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.float64),
    }


class _PresetEvaluator:
    def __init__(self, results: list[FitnessResult]) -> None:
        self._results = results
        self.last_programs: list[Any] | None = None

    def evaluate(
        self,
        programs: list[Any],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        del context
        self.last_programs = list(programs)
        return self._results[: len(programs)]


def _fr(value: float, *, stage: str = "") -> FitnessResult:
    return FitnessResult(objectives=(value,), metadata={"stage": stage})


class TestStage2StageAContract:
    def test_all_nan_scores_have_explicit_penalty_reason_code(self) -> None:
        evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        [result] = evaluator.evaluate(
            [_program()],
            _label_context(
                scores=[float("nan"), float("nan"), float("nan")],
                labels=[1.0, 0.0, 1.0],
            ),
        )
        assert result.objectives == (0.0,)
        assert result.metadata["stage_a_reason_code"] == "all_nan_scores"
        assert isinstance(result.metadata["stage_a_replay"], dict)

    def test_degenerate_scores_have_explicit_penalty_reason_code(self) -> None:
        evaluator = LabelFitnessEvaluator(metric="accuracy", top_k=0.5)
        [result] = evaluator.evaluate(
            [_program()],
            _label_context(
                scores=[0.2, 0.2, 0.2, 0.2],
                labels=[1.0, 0.0, 1.0, 0.0],
            ),
        )
        assert result.objectives == (0.0,)
        assert result.metadata["stage_a_reason_code"] == "degenerate_scores"
        assert result.metadata["reason"] == "degenerate_scores"

    def test_tie_handling_is_deterministic(self) -> None:
        evaluator = LabelFitnessEvaluator(metric="precision_at_k", top_k=0.5)
        context = _label_context(
            scores=[0.9, 0.9, 0.1, 0.1],
            labels=[1.0, 0.0, 1.0, 0.0],
        )
        [first] = evaluator.evaluate([_program()], context)
        [second] = evaluator.evaluate([_program()], context)
        assert first.objectives == second.objectives
        assert first.metadata["stage_a_reason_code"] == "ok"

    def test_empty_scores_have_explicit_penalty_reason_code(self) -> None:
        evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        [result] = evaluator.evaluate(
            [_program()],
            _label_context(scores=[], labels=[]),
        )
        assert result.objectives == (0.0,)
        assert result.metadata["stage_a_reason_code"] == "empty_scores"


class TestStage2PromotionGate:
    def test_tie_break_is_deterministic_and_stable_by_index(self) -> None:
        stage_a = _PresetEvaluator(
            [
                _fr(0.8, stage="a"),
                _fr(0.8, stage="a"),
                _fr(0.8, stage="a"),
                _fr(0.1, stage="a"),
            ]
        )
        stage_b = _PresetEvaluator([_fr(0.95, stage="b"), _fr(0.85, stage="b")])
        evaluator = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=stage_b, top_k=2)

        result = evaluator.evaluate(["p0", "p1", "p2", "p3"], {"x": np.array([1.0])})

        assert stage_b.last_programs == ["p0", "p1"]
        assert result[0].metadata["reason_code"] == "promoted_top_k"
        assert result[1].metadata["reason_code"] == "promoted_top_k"
        assert result[2].metadata["reason_code"] == "outside_stage_b_budget"

    def test_threshold_and_budget_controls_are_enforced(self) -> None:
        stage_a = _PresetEvaluator(
            [
                _fr(0.95, stage="a"),
                _fr(0.85, stage="a"),
                _fr(0.75, stage="a"),
                _fr(0.65, stage="a"),
            ]
        )
        stage_b = _PresetEvaluator([_fr(0.99, stage="b"), _fr(0.88, stage="b")])
        evaluator = TwoStageFitnessEvaluator(
            stage_a=stage_a,
            stage_b=stage_b,
            top_k=4,
            stage_b_candidate_budget=2,
            stage_a_threshold=0.7,
            stage_b_min_candidates=1,
        )

        result = evaluator.evaluate(
            ["p0", "p1", "p2", "p3"], {"x": np.array([1.0, 2.0])}
        )

        assert stage_b.last_programs == ["p0", "p1"]
        assert result[0].metadata["reason_code"] == "promoted_top_k"
        assert result[1].metadata["reason_code"] == "promoted_top_k"
        assert result[2].metadata["reason_code"] == "outside_stage_b_budget"
        assert result[3].metadata["reason_code"] == "below_stage_a_threshold"

    def test_min_candidates_fill_is_explicitly_reasoned(self) -> None:
        stage_a = _PresetEvaluator([_fr(0.9, stage="a"), _fr(0.6, stage="a"), _fr(0.4, stage="a")])
        stage_b = _PresetEvaluator([_fr(0.91, stage="b"), _fr(0.61, stage="b")])
        evaluator = TwoStageFitnessEvaluator(
            stage_a=stage_a,
            stage_b=stage_b,
            top_k=3,
            stage_a_threshold=0.95,
            stage_b_min_candidates=2,
        )

        result = evaluator.evaluate(["p0", "p1", "p2"], {"x": np.array([1.0])})
        assert stage_b.last_programs == ["p0", "p1"]
        assert result[0].metadata["reason_code"] == "promoted_min_candidates_fill"
        assert result[1].metadata["reason_code"] == "promoted_min_candidates_fill"
        assert result[2].metadata["reason_code"] == "below_stage_a_threshold"

    def test_gate_output_contains_lineage_and_reason_codes(self) -> None:
        stage_a = _PresetEvaluator([_fr(float("nan"), stage="a"), _fr(0.8, stage="a"), _fr(0.7, stage="a")])
        stage_b = _PresetEvaluator([_fr(0.85, stage="b"), _fr(0.75, stage="b")])
        evaluator = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=stage_b, top_k=2)

        result = evaluator.evaluate(["p0", "p1", "p2"], {"x": np.array([1.0])})
        assert result[0].metadata["reason_code"] == "stage_a_non_finite"
        assert result[1].metadata["two_stage_gate"]["lineage"]["final_stage"] == "stage_b"
        assert result[2].metadata["two_stage_gate"]["lineage"]["source_stage"] == "stage_a"

    def test_no_promotions_returns_annotated_stage_a_results(self) -> None:
        stage_a = _PresetEvaluator(
            [
                _fr(float("nan"), stage="a"),
                FitnessResult(objectives=(), metadata={"stage": "a"}),
            ]
        )
        stage_b = _PresetEvaluator([_fr(1.0, stage="b")])
        evaluator = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=stage_b, top_k=1)

        result = evaluator.evaluate(["p0", "p1"], {"x": np.array([1.0])})
        assert stage_b.last_programs is None
        assert result[0].metadata["reason_code"] == "stage_a_non_finite"
        assert result[1].metadata["reason_code"] == "stage_a_non_finite"

    def test_stage_a_result_count_mismatch_raises(self) -> None:
        stage_a = _PresetEvaluator([])
        evaluator = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=None, top_k=1)

        try:
            evaluator.evaluate(["p0"], {"x": np.array([1.0])})
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "Stage-A evaluator returned mismatched result count" in str(exc)

    def test_stage_b_result_count_mismatch_raises(self) -> None:
        stage_a = _PresetEvaluator([_fr(0.9, stage="a"), _fr(0.8, stage="a")])
        stage_b = _PresetEvaluator([])
        evaluator = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=stage_b, top_k=1)

        try:
            evaluator.evaluate(["p0", "p1"], {"x": np.array([1.0])})
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "Stage-B evaluator returned mismatched result count" in str(exc)

    def test_stage_b_zero_budget_branch(self) -> None:
        stage_a = _PresetEvaluator([_fr(0.9, stage="a"), _fr(0.8, stage="a")])
        stage_b = _PresetEvaluator([_fr(0.95, stage="b")])
        evaluator = TwoStageFitnessEvaluator(stage_a=stage_a, stage_b=stage_b, top_k=1)
        evaluator._stage_b_candidate_budget = 0  # force budget-zero path

        result = evaluator.evaluate(["p0", "p1"], {"x": np.array([1.0])})
        assert result[0].metadata["reason_code"] == "stage_b_budget_zero"
        assert result[1].metadata["reason_code"] == "stage_b_budget_zero"


class TestStage2GateEvaluatorDispatch:
    def test_stage_a_evaluate_fitness_fallback_is_supported(self) -> None:
        class _FitnessOnly:
            def evaluate_fitness(
                self,
                programs: list[Any],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                del context
                return [_fr(0.6, stage="a") for _ in programs]

        evaluator = TwoStageFitnessEvaluator(stage_a=_FitnessOnly(), stage_b=None, top_k=1)
        result = evaluator.evaluate(["p0"], {"x": np.array([1.0])})
        assert result[0].objectives == (0.6,)

    def test_stage_a_callable_fallback_is_supported(self) -> None:
        def _callable_stage_a(
            programs: list[Any], context: dict[str, np.ndarray]
        ) -> list[FitnessResult]:
            del context
            return [_fr(0.4, stage="a") for _ in programs]

        evaluator = TwoStageFitnessEvaluator(stage_a=_callable_stage_a, stage_b=None, top_k=1)
        result = evaluator.evaluate(["p0"], {"x": np.array([1.0])})
        assert result[0].objectives == (0.4,)

    def test_invalid_evaluator_type_raises(self) -> None:
        evaluator = TwoStageFitnessEvaluator(stage_a=object(), stage_b=None, top_k=1)
        try:
            evaluator.evaluate(["p0"], {"x": np.array([1.0])})
            raise AssertionError("expected TypeError")
        except TypeError as exc:
            assert "Evaluator must provide evaluate()" in str(exc)


class TestStage2GateValidation:
    def test_invalid_stage_b_candidate_budget_raises(self) -> None:
        try:
            TwoStageFitnessEvaluator(
                stage_a=_PresetEvaluator([]),
                stage_b=None,
                top_k=1,
                stage_b_candidate_budget=0,
            )
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "stage_b_candidate_budget" in str(exc)

    def test_invalid_stage_b_min_candidates_raises(self) -> None:
        try:
            TwoStageFitnessEvaluator(
                stage_a=_PresetEvaluator([]),
                stage_b=None,
                top_k=1,
                stage_b_min_candidates=0,
            )
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "stage_b_min_candidates" in str(exc)

    def test_invalid_stage_a_threshold_raises(self) -> None:
        try:
            TwoStageFitnessEvaluator(
                stage_a=_PresetEvaluator([]),
                stage_b=None,
                top_k=1,
                stage_a_threshold=0.0,
            )
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "stage_a_threshold" in str(exc)


class TestStage2ObjectiveWiring:
    def test_wire_objectives_passes_run_thresholds_to_two_stage_gate(self) -> None:
        fitness_cfg = FitnessStageConfig(use_backtest=True, backtest_top_n=7)
        run_cfg = EvolutionRunConfig(
            stage_b_candidate_budget=2,
            stage_b_min_candidates=2,
            stage_a_threshold=0.75,
        )

        evaluator = wire_objectives(
            fitness_cfg,
            backtest_fn=lambda _strategy: [],
            run_config=run_cfg,
        )
        assert isinstance(evaluator, TwoStageFitnessEvaluator)
        assert evaluator._top_k == 7
        assert evaluator._stage_b_candidate_budget == 2
        assert evaluator._stage_b_min_candidates == 2
        assert evaluator._stage_a_threshold == 0.75
