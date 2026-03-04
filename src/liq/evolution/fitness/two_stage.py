"""Two-stage fitness evaluation: fast screening followed by expensive refinement."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from liq.gp.types import FitnessResult
from liq.evolution.fitness.eval_cache import compute_program_hash

if TYPE_CHECKING:
    from liq.gp.program.ast import Program


logger = logging.getLogger(__name__)


class TwoStageFitnessEvaluator:
    """Two-stage fitness evaluator.

    Stage A evaluates all programs with a cheap metric (e.g. label-based).
    Stage B re-evaluates the top *top_k* programs with an expensive metric
    (e.g. backtest-based).  Programs outside the top K keep their Stage A
    results.

    When *stage_b* is ``None`` the evaluator acts as a passthrough for
    Stage A only.
    """

    def __init__(
        self,
        stage_a: Any,
        stage_b: Any | None = None,
        top_k: int = 10,
        *,
        stage_b_candidate_budget: int | None = None,
        stage_b_min_candidates: int = 1,
        stage_a_threshold: float | None = None,
    ) -> None:
        """Initialize two-stage evaluator.

        Args:
            stage_a: Primary evaluator applied to all programs.
            stage_b: Optional secondary evaluator for top K programs.
            top_k: Number of top programs to promote to Stage B.
            stage_b_candidate_budget: Optional hard cap for Stage-B promotions.
            stage_b_min_candidates: Minimum promotions to Stage B (if available).
            stage_a_threshold: Optional minimum Stage-A score required for direct
                promotion to Stage B.

        Raises:
            ValueError: If inputs are invalid.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if stage_b_candidate_budget is not None and stage_b_candidate_budget < 1:
            raise ValueError("stage_b_candidate_budget must be >= 1")
        if stage_b_min_candidates < 1:
            raise ValueError("stage_b_min_candidates must be >= 1")
        if stage_a_threshold is not None and not 0.0 < stage_a_threshold <= 1.0:
            raise ValueError("stage_a_threshold must be in (0, 1]")
        self._stage_a = stage_a
        self._stage_b = stage_b
        self._top_k = top_k
        self._stage_b_candidate_budget = stage_b_candidate_budget
        self._stage_b_min_candidates = stage_b_min_candidates
        self._stage_a_threshold = stage_a_threshold

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        """Evaluate programs through one or two stages.

        Args:
            programs: GP programs to evaluate.
            context: Evaluation context (features, labels, etc.).

        Returns:
            Fitness results for each program, in input order.
        """
        if not programs:
            return []

        context_map = dict(context or {})
        run_id = str(context_map.get("run_id", "unknown-run"))
        program_hashes = [self._program_hash(program) for program in programs]

        stage_a_start = time.perf_counter()
        try:
            stage_a_results = self._evaluate_with(self._stage_a, programs, context)
        except Exception:
            logger.exception(
                "two-stage stage_a_failed run_id=%s candidates=%s",
                run_id,
                len(programs),
            )
            raise
        stage_a_elapsed_ms = _to_ms(stage_a_start)
        if len(stage_a_results) != len(programs):
            raise ValueError("Stage-A evaluator returned mismatched result count")
        logger.info(
            "two-stage stage_a_complete run_id=%s candidates=%s elapsed_ms=%.3f",
            run_id,
            len(programs),
            stage_a_elapsed_ms,
        )

        if self._stage_b is None:
            _, reason_for_all = self._select_promoted(stage_a_results)
            for idx, reason_code in reason_for_all.items():
                logger.info(
                    "two-stage gate_decision run_id=%s candidate_hash=%s stage=%s reason=%s elapsed_ms=%.3f",
                    run_id,
                    program_hashes[idx],
                    "stage_a",
                    reason_code,
                    stage_a_elapsed_ms,
                )
            logger.info(
                "two-stage no_stage_b run_id=%s candidates=%s reason=%s elapsed_ms=%.3f",
                run_id,
                len(programs),
                "disabled",
                stage_a_elapsed_ms,
            )
            return [
                self._with_gate_metadata(
                    stage_a_result,
                    candidate_index=index,
                    stage_a_result=stage_a_result,
                    promoted=False,
                    reason_code=reason_code,
                    promotion_rank=None,
                )
                for index, (stage_a_result, reason_code) in enumerate(
                    zip(stage_a_results, reason_for_all.values())
                )
            ]

        selection_start = time.perf_counter()
        promoted, promotion_reason = self._select_promoted(stage_a_results)
        selection_elapsed_ms = _to_ms(selection_start)
        logger.info(
            "two-stage selection_complete run_id=%s candidates=%s promoted=%s elapsed_ms=%.3f",
            run_id,
            len(programs),
            len(promoted),
            selection_elapsed_ms,
        )

        for idx, reason_code in promotion_reason.items():
            logger.info(
                "two-stage gate_decision run_id=%s candidate_hash=%s stage=%s reason=%s elapsed_ms=%.3f",
                run_id,
                program_hashes[idx],
                "stage_b" if idx in set(promoted) else "stage_a",
                reason_code,
                selection_elapsed_ms,
            )

        if not promoted:
            return [
                self._with_gate_metadata(
                    stage_a_result,
                    candidate_index=idx,
                    stage_a_result=stage_a_result,
                    promoted=False,
                    reason_code=promotion_reason[idx],
                    promotion_rank=None,
                )
                for idx, stage_a_result in enumerate(stage_a_results)
            ]

        stage_b_programs = [programs[idx] for idx in promoted]
        stage_b_start = time.perf_counter()
        try:
            stage_b_results = self._evaluate_with(
                self._stage_b, stage_b_programs, context
            )
        except Exception:
            if promoted:
                candidate_hash = program_hashes[promoted[0]]
            else:
                candidate_hash = "unknown"
            logger.exception(
                "two-stage stage_b_failed run_id=%s candidate_hash=%s promoted=%s",
                run_id,
                candidate_hash,
                len(promoted),
            )
            raise
        stage_b_elapsed_ms = _to_ms(stage_b_start)
        logger.info(
            "two-stage stage_b_complete run_id=%s promoted=%s elapsed_ms=%.3f",
            run_id,
            len(promoted),
            stage_b_elapsed_ms,
        )
        if len(stage_b_results) != len(stage_b_programs):
            raise ValueError("Stage-B evaluator returned mismatched result count")

        promoted_rank = {idx: rank + 1 for rank, idx in enumerate(promoted)}
        merged: list[FitnessResult] = list(stage_a_results)
        for local_idx, program_idx in enumerate(promoted):
            reason_code = promotion_reason[program_idx]
            merged[program_idx] = self._with_gate_metadata(
                stage_b_results[local_idx],
                candidate_index=program_idx,
                stage_a_result=stage_a_results[program_idx],
                promoted=True,
                reason_code=reason_code,
                promotion_rank=promoted_rank[program_idx],
            )

        for idx, stage_a_result in enumerate(stage_a_results):
            if idx in promoted_rank:
                continue
            reason_code = promotion_reason[idx]
            merged[idx] = self._with_gate_metadata(
                stage_a_result,
                candidate_index=idx,
                stage_a_result=stage_a_result,
                promoted=False,
                reason_code=reason_code,
                promotion_rank=None,
            )

        return merged

    def evaluate_fitness(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        """Backward-compatible alias for :meth:`evaluate`.

        Args:
            programs: GP programs to evaluate.
            context: Evaluation context.

        Returns:
            Fitness results for each program.
        """
        return self.evaluate(programs, context)

    def _evaluate_with(
        self,
        evaluator: Any,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        if hasattr(evaluator, "evaluate") and callable(evaluator.evaluate):
            return evaluator.evaluate(programs, context)
        if hasattr(evaluator, "evaluate_fitness") and callable(
            evaluator.evaluate_fitness
        ):
            return evaluator.evaluate_fitness(programs, context)
        if callable(evaluator):
            return evaluator(programs, context)
        raise TypeError(
            "Evaluator must provide evaluate(), evaluate_fitness(), or be callable"
        )

    def _stage_a_score(self, result: FitnessResult) -> float:
        if not result.objectives:
            return float("nan")
        return float(result.objectives[0])

    def _effective_budget(self, population_size: int) -> int:
        budget = min(population_size, self._top_k)
        if self._stage_b_candidate_budget is not None:
            budget = min(budget, self._stage_b_candidate_budget)
        return budget

    def _select_promoted(
        self, stage_a_results: list[FitnessResult]
    ) -> tuple[list[int], dict[int, str]]:
        population = len(stage_a_results)
        if population == 0:
            return [], {}

        scores = [self._stage_a_score(result) for result in stage_a_results]
        finite_ranked = sorted(
            [
                idx
                for idx, score in enumerate(scores)
                if math.isfinite(score)
            ],
            key=lambda idx: (-scores[idx], idx),
        )
        reasons: dict[int, str] = {
            idx: (
                "stage_a_non_finite"
                if not math.isfinite(score)
                else "outside_stage_b_budget"
            )
            for idx, score in enumerate(scores)
        }

        budget = self._effective_budget(population)
        if budget <= 0:
            return [], {idx: "stage_b_budget_zero" for idx in range(population)}

        threshold_pass = finite_ranked
        if self._stage_a_threshold is not None:
            threshold_pass = [
                idx for idx in finite_ranked if scores[idx] >= self._stage_a_threshold
            ]

        promoted = threshold_pass[:budget]
        for idx in promoted:
            reasons[idx] = "promoted_top_k"

        required = min(self._stage_b_min_candidates, budget, len(finite_ranked))
        if len(promoted) < required:
            promoted_set = set(promoted)
            for idx in finite_ranked:
                if idx in promoted_set:
                    continue
                promoted.append(idx)
                promoted_set.add(idx)
                reasons[idx] = "promoted_min_candidates_fill"
                if len(promoted) >= required:
                    break

        if self._stage_a_threshold is not None:
            below = {idx for idx in finite_ranked if idx not in threshold_pass}
            for idx in below:
                if idx not in promoted:
                    reasons[idx] = "below_stage_a_threshold"

        return promoted, reasons

    def _with_gate_metadata(
        self,
        result: FitnessResult,
        *,
        candidate_index: int,
        stage_a_result: FitnessResult,
        promoted: bool,
        reason_code: str,
        promotion_rank: int | None,
    ) -> FitnessResult:
        base_metadata = dict(result.metadata)
        score = self._stage_a_score(stage_a_result)
        gate_metadata = {
            "reason_code": reason_code,
            "promoted_to_stage_b": promoted,
            "promotion_rank": promotion_rank,
            "candidate_index": candidate_index,
            "stage_a_score": score if math.isfinite(score) else None,
            "gate_config": {
                "top_k": self._top_k,
                "stage_b_candidate_budget": self._stage_b_candidate_budget,
                "stage_b_min_candidates": self._stage_b_min_candidates,
                "stage_a_threshold": self._stage_a_threshold,
            },
            "lineage": {
                "source_stage": "stage_a",
                "final_stage": "stage_b" if promoted else "stage_a",
            },
        }
        base_metadata["two_stage_gate"] = gate_metadata
        base_metadata["reason_code"] = reason_code
        return FitnessResult(objectives=result.objectives, metadata=base_metadata)

    def _program_hash(self, program: Any) -> str:
        try:
            return compute_program_hash(program)
        except Exception:
            payload = str(program)
            if not payload:
                payload = "<empty-program>"
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _to_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
