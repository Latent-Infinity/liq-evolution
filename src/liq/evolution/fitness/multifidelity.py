"""General-purpose multi-fidelity evaluator with direction-aware promotion."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from liq.evolution.fitness.evaluation_schema import ObjectiveDirection
from liq.gp.types import FitnessResult

PromotionStrategy = Literal[
    "direction_aware_first",
    "scalarized",
    "pareto",
    "niche",
]


class MultiFidelityFitnessEvaluator:
    """Evaluate programs across multiple fidelity levels.

    Levels are evaluated sequentially. Every level is a full evaluator object with an
    ``evaluate()`` method. After each level, a subset of candidates is promoted to
    the next level according to a strategy.
    """

    evaluator_version = "1.0"

    def __init__(
        self,
        levels: Sequence[Any],
        *,
        top_k_per_level: int | float | Sequence[int | float] = 1,
        objective_directions: Sequence[ObjectiveDirection] = ("maximize",),
        promotion_strategy: PromotionStrategy = "direction_aware_first",
        scalarization_weights: Sequence[float] | None = None,
        niche_metadata_key: str = "archive_bin",
        level_costs: Sequence[float] | None = None,
        fallback_niche_metadata_keys: Sequence[str] = ("qd_bin", "niche", "bin"),
    ) -> None:
        if not levels:
            raise ValueError("levels must contain at least one evaluator")

        if promotion_strategy not in {
            "direction_aware_first",
            "scalarized",
            "pareto",
            "niche",
        }:
            raise ValueError(
                "promotion_strategy must be one of:"
                " direction_aware_first, scalarized, pareto, niche"
            )

        self._levels = list(levels)
        self._promotion_strategy = promotion_strategy
        self._objective_directions = tuple(objective_directions)
        self._niche_metadata_key = niche_metadata_key
        self._fallback_niche_metadata_keys = tuple(fallback_niche_metadata_keys)

        if not self._objective_directions:
            raise ValueError("objective_directions must be non-empty")

        for direction in self._objective_directions:
            if direction not in {"maximize", "minimize"}:
                raise ValueError(
                    "objective_directions must be 'maximize' or 'minimize'"
                )

        if scalarization_weights is not None and any(
            weight < 0.0 for weight in scalarization_weights
        ):
            raise ValueError("scalarization_weights must be >= 0")
        self._scalarization_weights = (
            tuple(scalarization_weights) if scalarization_weights is not None else None
        )

        if len(self._levels) > 1:
            self._top_k_per_transition = self._normalize_top_k(
                top_k_per_level,
                len(self._levels) - 1,
            )
        else:
            self._top_k_per_transition = []

        if level_costs is not None:
            self._level_costs = tuple(float(value) for value in level_costs)
            if len(self._level_costs) != len(self._levels):
                raise ValueError("level_costs must have one entry per level")
            if any(cost < 0.0 for cost in self._level_costs):
                raise ValueError("level_costs must be non-negative")
            for prev, nxt in zip(
                self._level_costs, self._level_costs[1:], strict=False
            ):
                if nxt < prev:
                    raise ValueError("level_costs must be non-decreasing")
        else:
            self._level_costs = tuple(float(i + 1) for i in range(len(self._levels)))

        self._fingerprint_cache: str | None = None

    @property
    def level_costs(self) -> tuple[float, ...]:
        """Return per-level monotonic cost multipliers used for accounting."""
        return self._level_costs

    @property
    def evaluator_fingerprint(self) -> str:
        """Deterministic fingerprint for evaluator configuration and version."""
        if self._fingerprint_cache is None:
            payload = {
                "version": self.evaluator_version,
                "top_k_per_level": self._top_k_per_transition,
                "objective_directions": self._objective_directions,
                "promotion_strategy": self._promotion_strategy,
                "scalarization_weights": self._scalarization_weights,
                "niche_metadata_key": self._niche_metadata_key,
                "fallback_niche_metadata_keys": self._fallback_niche_metadata_keys,
                "level_count": len(self._levels),
                "level_costs": self._level_costs,
            }
            raw = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), default=str
            )
            self._fingerprint_cache = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self._fingerprint_cache

    @staticmethod
    def _normalize_top_k(
        top_k: int | float | Sequence[int | float],
        expected: int,
    ) -> list[int | float]:
        if expected <= 0:
            return []

        if isinstance(top_k, (int, float)):
            normalized = [top_k] * expected
        else:
            normalized = list(top_k)
            if len(normalized) != expected:
                raise ValueError(
                    "top_k_per_level must be a scalar or one value per transition"
                )

        for value in normalized:
            if isinstance(value, bool):
                raise ValueError("top_k_per_level values must be numeric")
            numeric = float(value)
            if numeric <= 0:
                raise ValueError("top_k_per_level values must be > 0")
            if numeric > 1.0 and isinstance(value, float):
                raise ValueError(
                    "top_k_per_level fractions must be in (0.0, 1.0] when float"
                )
            if not math.isfinite(numeric):
                raise ValueError("top_k_per_level values must be finite")
        return normalized

    def evaluate(
        self,
        programs: list[Any],
        context: Mapping[str, Any],
    ) -> list[FitnessResult]:
        if not programs:
            return []

        results = self._evaluate_level(0, programs, context)

        active_indices = list(range(len(programs)))
        for level_idx in range(1, len(self._levels)):
            if not active_indices:
                break

            budget = self._resolve_budget(
                self._top_k_per_transition[level_idx - 1],
                len(active_indices),
            )
            if budget <= 0:
                break

            promote = self._select_promoted_indices(
                active_indices,
                results,
                budget,
                level_idx,
            )
            if not promote:
                break

            to_evaluate = [programs[i] for i in promote]
            next_results = self._evaluate_level(level_idx, to_evaluate, context)
            if len(next_results) != len(to_evaluate):
                raise ValueError(
                    "Evaluator returned mismatched result count for promoted subset"
                )

            for local_idx, program_idx in enumerate(promote):
                results[program_idx] = next_results[local_idx]

            active_indices = promote

        return results

    def evaluate_fitness(
        self,
        programs: list[Any],
        context: Mapping[str, Any],
    ) -> list[FitnessResult]:
        return self.evaluate(programs, context)

    def _evaluate_level(
        self,
        level: int,
        programs: list[Any],
        context: Mapping[str, Any],
    ) -> list[FitnessResult]:
        evaluator = self._levels[level]
        if hasattr(evaluator, "evaluate") and callable(evaluator.evaluate):
            return evaluator.evaluate(programs, context)
        if hasattr(evaluator, "evaluate_fitness") and callable(
            evaluator.evaluate_fitness
        ):
            return evaluator.evaluate_fitness(programs, context)
        if callable(evaluator):
            return evaluator(programs, context)
        raise TypeError(
            "Level evaluator must provide evaluate(), evaluate_fitness(), or be callable"
        )

    def _resolve_budget(
        self,
        raw_top_k: int | float,
        population_count: int,
    ) -> int:
        if population_count <= 0:
            return 0
        if isinstance(raw_top_k, int):
            count = int(raw_top_k)
        else:
            ratio = float(raw_top_k)
            if ratio >= 1.0:
                count = population_count
            else:
                count = max(1, math.ceil(population_count * ratio))
        return min(population_count, count)

    def _select_promoted_indices(
        self,
        candidate_indices: list[int],
        results: list[FitnessResult],
        budget: int,
        level: int,
    ) -> list[int]:
        del level
        if self._promotion_strategy == "direction_aware_first":
            return self._select_by_direction_aware_first(
                candidate_indices, results, budget
            )
        if self._promotion_strategy == "scalarized":
            return self._select_by_scalarized_score(candidate_indices, results, budget)
        if self._promotion_strategy == "pareto":
            return self._select_by_pareto(candidate_indices, results, budget)
        return self._select_by_niche(candidate_indices, results, budget)

    def _select_by_direction_aware_first(
        self,
        candidate_indices: list[int],
        results: list[FitnessResult],
        budget: int,
    ) -> list[int]:
        scored = [
            (self._rank_key(results[idx], 0, idx), idx) for idx in candidate_indices
        ]
        scored.sort(key=lambda item: item[0])
        return [idx for _, idx in scored[:budget]]

    def _select_by_scalarized_score(
        self,
        candidate_indices: list[int],
        results: list[FitnessResult],
        budget: int,
    ) -> list[int]:
        scored: list[tuple[tuple[float, float, int], int]] = []
        for idx in candidate_indices:
            result = results[idx]
            score = 0.0
            bad = False
            for obj_idx, direction in enumerate(self._objective_directions):
                value = self._objective_value(result, obj_idx)
                if not math.isfinite(value):
                    bad = True
                    break

                weight = 1.0
                if self._scalarization_weights is not None and obj_idx < len(
                    self._scalarization_weights
                ):
                    weight = float(self._scalarization_weights[obj_idx])

                oriented = value if direction == "maximize" else -value
                score += oriented * weight
            ranked = (1.0, 0.0, idx) if bad else (0.0, -score, idx)
            scored.append((ranked, idx))

        scored.sort(key=lambda item: item[0])
        return [idx for _, idx in scored[:budget]]

    def _select_by_pareto(
        self,
        candidate_indices: list[int],
        results: list[FitnessResult],
        budget: int,
    ) -> list[int]:
        if budget <= 0:
            return []

        selected: list[int] = []
        remaining = list(candidate_indices)

        while remaining and len(selected) < budget:
            front = self._pareto_front(remaining, results)
            if len(selected) + len(front) <= budget:
                ordered_front = self._sort_by_crowding(front, results)
                selected.extend(ordered_front)
                remaining = [idx for idx in remaining if idx not in set(front)]
                continue

            ordered_front = self._sort_by_crowding(front, results)
            selected.extend(ordered_front[: budget - len(selected)])
            break

        return selected[:budget]

    def _select_by_niche(
        self,
        candidate_indices: list[int],
        results: list[FitnessResult],
        budget: int,
    ) -> list[int]:
        if budget <= 0:
            return []

        groups: dict[str, list[int]] = defaultdict(list)
        for idx in candidate_indices:
            key = self._extract_niche(results[idx])
            groups[key].append(idx)

        per_niche = max(1, math.ceil(budget / max(1, len(groups))))
        selected: list[tuple[tuple[float, float, int], int]] = []
        for _, candidate_group in sorted(groups.items()):
            scored = [
                (self._rank_key(results[idx], 0, idx), idx) for idx in candidate_group
            ]
            scored.sort(key=lambda item: item[0])
            selected.extend(scored[:per_niche])

        selected.sort(key=lambda item: item[0])
        return [idx for _, idx in selected[:budget]]

    def _pareto_front(
        self,
        indices: list[int],
        results: list[FitnessResult],
    ) -> list[int]:
        front: list[int] = []
        for i in indices:
            dominated = False
            for j in indices:
                if i == j:
                    continue
                if self._dominates(j, i, results):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        return front

    def _dominates(self, left: int, right: int, results: list[FitnessResult]) -> bool:
        left_res = results[left]
        right_res = results[right]
        better_in_any = False

        for obj_idx, direction in enumerate(self._objective_directions):
            lval = self._objective_value(left_res, obj_idx)
            rval = self._objective_value(right_res, obj_idx)
            if not math.isfinite(lval) and not math.isfinite(rval):
                continue
            if not math.isfinite(lval):
                return False
            if not math.isfinite(rval):
                better_in_any = True
                continue

            if direction == "maximize":
                if lval < rval:
                    return False
                if lval > rval:
                    better_in_any = True
            else:
                if lval > rval:
                    return False
                if lval < rval:
                    better_in_any = True

        return better_in_any

    def _sort_by_crowding(
        self,
        indices: list[int],
        results: list[FitnessResult],
    ) -> list[int]:
        if len(indices) <= 2:
            return sorted(indices)

        crowding: dict[int, float] = dict.fromkeys(indices, 0.0)
        for obj_idx, direction in enumerate(self._objective_directions):
            finite: list[tuple[int, float]] = []
            for idx in indices:
                value = self._objective_value(results[idx], obj_idx)
                if math.isfinite(value):
                    oriented = value if direction == "maximize" else -value
                    finite.append((idx, oriented))

            finite.sort(key=lambda item: item[1])
            if len(finite) < 2:
                continue

            lo = finite[0][1]
            hi = finite[-1][1]
            range_width = hi - lo
            if range_width <= 0.0:
                continue

            for idx in (finite[0][0], finite[-1][0]):
                crowding[idx] = float("inf")

            for left_idx, right_idx in zip(finite[1:-1], finite[2:], strict=False):
                mid = left_idx[0]
                crowding[mid] += (right_idx[1] - left_idx[1]) / range_width

        sorted_indices = sorted(
            indices,
            key=lambda idx: (
                0.0 if math.isinf(crowding[idx]) else 1.0,
                -crowding[idx] if not math.isinf(crowding[idx]) else 0.0,
                idx,
            ),
        )
        return sorted_indices

    def _rank_key(
        self, result: FitnessResult, objective_idx: int, index: int
    ) -> tuple[float, float, int]:
        if len(self._objective_directions) <= objective_idx:
            direction = "maximize"
        else:
            direction = self._objective_directions[objective_idx]

        raw_value = self._objective_value(result, objective_idx)
        if not math.isfinite(raw_value):
            return (1.0, 0.0, index)

        key_value = -raw_value if direction == "maximize" else raw_value
        return (0.0, key_value, index)

    def _objective_value(self, result: FitnessResult, idx: int) -> float:
        if idx >= len(result.objectives):
            return float("nan")

        value = result.objectives[idx]
        return float(value)

    def _extract_niche(self, result: FitnessResult) -> str:
        metadata = result.metadata or {}
        for key in (
            self._niche_metadata_key,
            *self._fallback_niche_metadata_keys,
        ):
            if key in metadata:
                return str(metadata[key])
        return "__all__"
