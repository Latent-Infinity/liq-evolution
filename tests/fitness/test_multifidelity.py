"""Tests for multi-fidelity evaluator promotion and budgets."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from liq.evolution.fitness.multifidelity import MultiFidelityFitnessEvaluator
from liq.gp.types import FitnessResult


class _ScoreByProgramEvaluator:
    """Simple evaluator returning preconfigured scores per program key."""

    def __init__(
        self,
        scores: Sequence[float],
        *,
        metadata: dict[int, object] | None = None,
        objectives_per_program: int = 1,
    ) -> None:
        self._scores = tuple(scores)
        self._metadata = metadata or {}
        self.calls: list[tuple[object, ...]] = []

        if objectives_per_program < 1:
            raise ValueError("objectives_per_program must be >= 1")
        self._objectives_per_program = objectives_per_program

    def evaluate(
        self, programs: Sequence[int], context: dict[str, object]
    ) -> list[FitnessResult]:
        del context
        self.calls.append(tuple(programs))
        results: list[FitnessResult] = []
        for program in programs:
            base_offset = program % len(self._scores)
            score = float(self._scores[base_offset])
            objectives = (
                (score,) if self._objectives_per_program == 1 else (score, score / 2)
            )
            metadata = {
                "archive_bin": "default_bin",
                "raw_objectives": objectives,
            }
            metadata.update(self._metadata)
            results.append(FitnessResult(objectives=objectives, metadata=metadata))
        return results


class TestMultiFidelityEvaluation:
    """Promotion and level-budget behavior."""

    def test_level0_evaluates_every_program(self) -> None:
        level0 = _ScoreByProgramEvaluator([0.1, 0.9, 0.5, 0.7])
        level1 = _ScoreByProgramEvaluator([3.0, 3.1, 3.2, 3.3])
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=0.5,
            objective_directions=("maximize",),
            promotion_strategy="direction_aware_first",
            level_costs=(1.0, 2.0),
        )

        result = evaluator.evaluate([0, 1, 2, 3], {})

        assert len(level0.calls) == 1
        assert set(level0.calls[0]) == {0, 1, 2, 3}
        assert level1.calls == [(1, 3)]  # top two by objective
        assert result[1].objectives == (3.1,)
        assert result[3].objectives == (3.3,)
        assert result[0].objectives == (0.1,)

    def test_direction_aware_first_uses_direction(self) -> None:
        level0 = _ScoreByProgramEvaluator([5.0, 1.0, 3.0])
        level1 = _ScoreByProgramEvaluator([9.0, 8.0, 7.0])
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=1,
            objective_directions=("minimize",),
            promotion_strategy="direction_aware_first",
        )

        evaluator.evaluate([0, 1, 2], {})

        assert level1.calls == [(1,)]  # lowest objective first for minimize

    def test_ties_are_deterministic_by_index(self) -> None:
        level0 = _ScoreByProgramEvaluator([0.5, 0.5, 0.2])
        level1 = _ScoreByProgramEvaluator([10.0, 11.0, 12.0])
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=1 / 3,
            objective_directions=("maximize",),
            promotion_strategy="direction_aware_first",
        )

        evaluator.evaluate([0, 1, 2], {})
        assert level1.calls == [(0,)]  # lower index wins ties

    def test_all_tied_population_is_deterministic_by_index(self) -> None:
        level0 = _ScoreByProgramEvaluator([0.5, 0.5, 0.5])
        level1 = _ScoreByProgramEvaluator([8.0, 9.0, 10.0])
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=2 / 3,
            objective_directions=("maximize",),
            promotion_strategy="direction_aware_first",
        )

        evaluator.evaluate([0, 1, 2], {})
        assert level1.calls == [(0, 1)]

    def test_nan_objectives_are_demoted_in_direction_aware_first(self) -> None:
        class _NaNDirectionEvaluator:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                outputs: list[tuple[float]] = []
                for program in programs:
                    if int(program) == 0:
                        outputs.append((float("nan"),))
                    elif int(program) == 1:
                        outputs.append((0.8,))
                    else:
                        outputs.append((0.7,))
                return [FitnessResult(objectives=item, metadata={}) for item in outputs]

        level0 = _NaNDirectionEvaluator()
        level1 = _ScoreByProgramEvaluator([9.0, 8.0, 7.0])
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=1 / 3,
            objective_directions=("maximize",),
            promotion_strategy="direction_aware_first",
        )

        evaluator.evaluate([0, 1, 2], {})
        assert level1.calls == [(1,)]

    def test_nan_objectives_are_demoted_in_scalarized_strategy(self) -> None:
        class _NaNFrontEvaluator:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                outputs = []
                for program in programs:
                    if program == 0:
                        objectives = (0.9, float("nan"))
                    elif program == 1:
                        objectives = (0.5, 1.0)
                    else:
                        objectives = (0.7, -0.2)
                    outputs.append(FitnessResult(objectives=objectives, metadata={}))
                return outputs

        level0 = _NaNFrontEvaluator()
        level1 = _ScoreByProgramEvaluator([5.0, 4.0, 3.0])
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=2 / 3,
            objective_directions=("maximize", "minimize"),
            promotion_strategy="scalarized",
        )

        evaluator.evaluate([0, 1, 2], {})
        assert level1.calls == [
            (2, 1)
        ]  # finite objectives sorted before NaN-missing case

    def test_nan_objectives_are_demoted_in_pareto(self) -> None:
        class _ParetoNaNEvaluator:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                score_map = {
                    0: (float("nan"),),
                    1: (1.0, 1.0),
                    2: (0.9, float("nan")),
                }
                return [
                    FitnessResult(
                        objectives=score_map[int(program)],
                        metadata={},
                    )
                    for program in programs
                ]

        level0 = _ParetoNaNEvaluator()
        level1 = _ScoreByProgramEvaluator([2.0, 1.0, 3.0], objectives_per_program=1)
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=2 / 3,
            objective_directions=("maximize", "maximize"),
            promotion_strategy="pareto",
        )

        evaluator.evaluate([0, 1, 2], {})
        assert set(level1.calls[0]) == {1, 2}

    def test_nan_objectives_are_demoted_in_niche(self) -> None:
        class _NicheNaNEvaluator:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                score_map = {
                    0: (float("nan"),),
                    1: (1.0,),
                    2: (0.3,),
                    3: (0.9,),
                }
                return [
                    FitnessResult(
                        objectives=score_map[int(program)],
                        metadata={
                            "archive_bin": "bin_a" if int(program) % 2 == 0 else "bin_b"
                        },
                    )
                    for program in programs
                ]

        level0 = _NicheNaNEvaluator()
        level1 = _ScoreByProgramEvaluator([3.0, 4.0, 5.0, 6.0])
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=1 / 4,
            objective_directions=("maximize",),
            promotion_strategy="niche",
            fallback_niche_metadata_keys=("archive_bin",),
        )

        evaluator.evaluate([0, 1, 2, 3], {})
        assert level1.calls == [(1,)]

    def test_short_objective_vectors_are_padded_with_nan(self) -> None:
        class _ShortObjectiveEvaluator:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                score_map = {
                    0: (0.4,),
                    1: (0.9, 0.0),
                    2: (0.1, 1.0),
                }
                return [
                    FitnessResult(
                        objectives=score_map[int(program)],
                        metadata={},
                    )
                    for program in programs
                ]

        level0 = _ShortObjectiveEvaluator()
        level1 = _ScoreByProgramEvaluator([9.0, 8.0, 7.0])
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=1 / 3,
            objective_directions=("maximize", "minimize"),
            promotion_strategy="scalarized",
        )

        evaluator.evaluate([0, 1, 2], {})
        assert level1.calls == [(1,)]

    def test_pareto_promotion_respects_front(self) -> None:
        class _ParetoEvaluator:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                score_map = {
                    0: (1.0, 0.0),
                    1: (0.0, 1.0),
                    2: (0.8, 0.8),
                }
                return [
                    FitnessResult(
                        objectives=score_map[int(program)],
                        metadata={},
                    )
                    for program in programs
                ]

        level0 = _ParetoEvaluator()
        level1 = _ScoreByProgramEvaluator([2.0, 1.0, 3.0], objectives_per_program=1)
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=2 / 3,
            objective_directions=("maximize", "maximize"),
            promotion_strategy="pareto",
        )

        evaluator.evaluate([0, 1, 2], {})
        assert level1.calls == [(0, 1)]

    def test_niche_promotion_selects_per_metadata_bin(self) -> None:
        class _BinEvaluator:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                return [
                    FitnessResult(
                        objectives=(score,),
                        metadata={
                            "archive_bin": "bin_a" if int(program) % 2 else "bin_b"
                        },
                    )
                    for program, score in zip(
                        programs,
                        (9.0, 2.0, 8.0, 1.0),
                        strict=False,
                    )
                ]

        level0 = _BinEvaluator()
        level1 = _ScoreByProgramEvaluator([3.0, 4.0, 5.0, 6.0])

        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=1.0,
            objective_directions=("maximize",),
            promotion_strategy="niche",
            fallback_niche_metadata_keys=("archive_bin",),
        )

        evaluator.evaluate([0, 1, 2, 3], {})
        assert set(level1.calls[0]) == {0, 1, 2, 3}

    def test_level_costs_are_exposed_and_monotone(self) -> None:
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(_ScoreByProgramEvaluator([1.0]), _ScoreByProgramEvaluator([2.0])),
            top_k_per_level=1,
            objective_directions=("maximize",),
            level_costs=(0.5, 0.9),
        )

        assert evaluator.level_costs == (0.5, 0.9)

        with pytest.raises(ValueError, match="level_costs must be non-decreasing"):
            MultiFidelityFitnessEvaluator(
                levels=(
                    _ScoreByProgramEvaluator([1.0]),
                    _ScoreByProgramEvaluator([2.0]),
                ),
                top_k_per_level=1,
                level_costs=(2.0, 1.0),
            )

    def test_invalid_initialization(self) -> None:
        with pytest.raises(
            ValueError, match="levels must contain at least one evaluator"
        ):
            MultiFidelityFitnessEvaluator(levels=())

        with pytest.raises(ValueError, match="promotion_strategy must be one of"):
            MultiFidelityFitnessEvaluator(
                levels=(_ScoreByProgramEvaluator([1.0]),),
                promotion_strategy="random",  # type: ignore[arg-type]
            )

        with pytest.raises(ValueError, match="objective_directions must be non-empty"):
            MultiFidelityFitnessEvaluator(
                levels=(_ScoreByProgramEvaluator([1.0]),),
                objective_directions=(),
            )

        with pytest.raises(
            ValueError, match="objective_directions must be 'maximize' or 'minimize'"
        ):
            MultiFidelityFitnessEvaluator(
                levels=(_ScoreByProgramEvaluator([1.0]),),
                objective_directions=("invalid",),  # type: ignore[arg-type]
            )

        with pytest.raises(ValueError, match="scalarization_weights must be >= 0"):
            MultiFidelityFitnessEvaluator(
                levels=(_ScoreByProgramEvaluator([1.0]),),
                scalarization_weights=(-0.1,),
            )

        with pytest.raises(ValueError, match="top_k_per_level values must be > 0"):
            MultiFidelityFitnessEvaluator(
                levels=(
                    _ScoreByProgramEvaluator([1.0]),
                    _ScoreByProgramEvaluator([2.0]),
                ),
                top_k_per_level=0,
            )

        with pytest.raises(ValueError, match="top_k_per_level fractions must be in"):
            MultiFidelityFitnessEvaluator(
                levels=(
                    _ScoreByProgramEvaluator([1.0]),
                    _ScoreByProgramEvaluator([2.0]),
                ),
                top_k_per_level=1.5,
            )

    def test_evaluator_fingerprint_is_deterministic(self) -> None:
        evaluator_a = MultiFidelityFitnessEvaluator(
            levels=(_ScoreByProgramEvaluator([1.0]), _ScoreByProgramEvaluator([2.0])),
            top_k_per_level=0.5,
            objective_directions=("maximize",),
            promotion_strategy="scalarized",
            scalarization_weights=(1.0,),
            fallback_niche_metadata_keys=("qd_bin",),
            level_costs=(1.0, 2.0),
        )
        evaluator_b = MultiFidelityFitnessEvaluator(
            levels=(_ScoreByProgramEvaluator([1.0]), _ScoreByProgramEvaluator([2.0])),
            top_k_per_level=0.5,
            objective_directions=("maximize",),
            promotion_strategy="scalarized",
            scalarization_weights=(1.0,),
            fallback_niche_metadata_keys=("qd_bin",),
            level_costs=(1.0, 2.0),
        )
        evaluator_c = MultiFidelityFitnessEvaluator(
            levels=(_ScoreByProgramEvaluator([1.0]), _ScoreByProgramEvaluator([2.0])),
            top_k_per_level=0.5,
            objective_directions=("maximize",),
            promotion_strategy="scalarized",
            scalarization_weights=(2.0,),
            fallback_niche_metadata_keys=("qd_bin",),
            level_costs=(1.0, 2.0),
        )

        assert evaluator_a.evaluator_fingerprint == evaluator_b.evaluator_fingerprint
        assert evaluator_a.evaluator_fingerprint != evaluator_c.evaluator_fingerprint

    def test_evaluate_empty_programs_returns_empty(self) -> None:
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(_ScoreByProgramEvaluator([1.0]),),
            top_k_per_level=1,
            objective_directions=("maximize",),
        )

        assert evaluator.evaluate([], {}) == []

    def test_evaluate_fitness_alias_and_level_type_paths(self) -> None:
        class _EvaluateFitnessOnly:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def evaluate_fitness(
                self,
                programs: Sequence[int],
                context: dict[str, object],
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                return [
                    FitnessResult(objectives=(float(program),), metadata={})
                    for program in programs
                ]

        class _CallableLevel:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def __call__(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                return [
                    FitnessResult(objectives=(float(program),), metadata={})
                    for program in programs
                ]

        eval_fitness_only = _EvaluateFitnessOnly()
        callable_level = _CallableLevel()

        evaluator = MultiFidelityFitnessEvaluator(
            levels=(eval_fitness_only, callable_level),
            top_k_per_level=1 / 2,
            objective_directions=("maximize",),
            promotion_strategy="direction_aware_first",
        )
        results = evaluator.evaluate_fitness([0, 1], {})

        assert results[0].objectives == (0.0,)
        assert eval_fitness_only.calls == [(0, 1)]
        assert callable_level.calls == [(1,)]

        with pytest.raises(TypeError, match="Level evaluator must provide evaluate"):
            evaluator_invalid = MultiFidelityFitnessEvaluator(
                levels=(object(),),
                top_k_per_level=1,
                objective_directions=("maximize",),
            )
            evaluator_invalid.evaluate([0], {})

    def test_zero_or_negative_budget_paths(self) -> None:
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(_ScoreByProgramEvaluator([1.0]),),
            top_k_per_level=1,
            objective_directions=("maximize",),
        )

        assert evaluator._resolve_budget(0.5, 0) == 0
        assert evaluator._resolve_budget(0, 10) == 0

    def test_pareto_short_crowding_branches(self) -> None:
        class _SmallFrontEvaluator:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                scores = {
                    0: (1.0, 1.0),
                    1: (1.0, 0.5),
                    2: (0.9, 1.0),
                }
                return [
                    FitnessResult(
                        objectives=scores[int(program)],
                        metadata={},
                    )
                    for program in programs
                ]

        level0 = _SmallFrontEvaluator()
        level1 = _ScoreByProgramEvaluator([9.0, 8.0, 7.0], objectives_per_program=1)

        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=2 / 3,
            objective_directions=("maximize", "maximize"),
            promotion_strategy="pareto",
        )
        evaluator.evaluate([0, 1, 2], {})
        assert level1.calls == [(0, 1)]

    def test_pareto_dominance_with_nan_and_minimize_edge_cases(self) -> None:
        class _DominanceEvaluator:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                scores = {
                    0: (1.0, float("nan")),
                    1: (0.5, 0.2),
                    2: (float("nan"), float("nan")),
                    3: (1.2, float("inf")),
                }
                return [
                    FitnessResult(
                        objectives=scores[int(program)],
                        metadata={},
                    )
                    for program in programs
                ]

        level0 = _DominanceEvaluator()
        level1 = _ScoreByProgramEvaluator(
            [2.0, 1.0, 0.0, 3.0], objectives_per_program=1
        )

        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=3 / 4,
            objective_directions=("maximize", "minimize"),
            promotion_strategy="pareto",
        )
        evaluator.evaluate([0, 1, 2, 3], {})
        assert set(level1.calls[0]) == {0, 1, 3}

    def test_niche_falls_back_to_default_bin(self) -> None:
        class _NoNicheMetadataEvaluator:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                return [
                    FitnessResult(objectives=(1.0 - 0.2 * int(program),), metadata={})
                    for program in programs
                ]

        level0 = _NoNicheMetadataEvaluator()
        level1 = _ScoreByProgramEvaluator([3.0, 4.0, 5.0], objectives_per_program=1)

        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=1 / 2,
            objective_directions=("maximize",),
            promotion_strategy="niche",
            fallback_niche_metadata_keys=("missing",),
        )

        evaluator.evaluate([0, 1, 2], {})
        assert level1.calls == [(0, 1)]

    def test_rank_key_prefers_default_objective_direction_when_missing(self) -> None:
        level0 = _ScoreByProgramEvaluator([1.0])
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0,),
            top_k_per_level=1,
            objective_directions=("maximize",),
        )
        result = FitnessResult(objectives=(1.0,), metadata={})

        assert evaluator._rank_key(result, 1, 0)[0] == 1.0

    def test_normalize_top_k_validation_edges(self) -> None:
        with pytest.raises(ValueError, match="scalar or one value per transition"):
            MultiFidelityFitnessEvaluator._normalize_top_k((1, 2), 1)

        with pytest.raises(ValueError, match="values must be numeric"):
            MultiFidelityFitnessEvaluator._normalize_top_k(True, 1)

        with pytest.raises(ValueError, match="fractions must be in"):
            MultiFidelityFitnessEvaluator._normalize_top_k(float("inf"), 1)

        assert MultiFidelityFitnessEvaluator._normalize_top_k([0.5], 0) == []
        assert MultiFidelityFitnessEvaluator._normalize_top_k([], 0) == []

    def test_level_cost_validation_paths(self) -> None:
        with pytest.raises(
            ValueError, match="level_costs must have one entry per level"
        ):
            MultiFidelityFitnessEvaluator(
                levels=(
                    _ScoreByProgramEvaluator([1.0]),
                    _ScoreByProgramEvaluator([2.0]),
                ),
                top_k_per_level=0.5,
                level_costs=(1.0,),
            )

        with pytest.raises(ValueError, match="level_costs must be non-negative"):
            MultiFidelityFitnessEvaluator(
                levels=(
                    _ScoreByProgramEvaluator([1.0]),
                    _ScoreByProgramEvaluator([2.0]),
                ),
                top_k_per_level=0.5,
                level_costs=(-1.0, 2.0),
            )

    def test_evaluate_control_flow_breaks(self) -> None:
        class _MismatchedEvaluator:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                return (
                    [FitnessResult(objectives=(1.0,), metadata={})] if programs else []
                )

        level0 = _ScoreByProgramEvaluator([1.0, 2.0])
        level1 = _MismatchedEvaluator()
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=1,
            objective_directions=("maximize",),
            promotion_strategy="direction_aware_first",
        )

        evaluator._top_k_per_transition = [2]
        with pytest.raises(
            ValueError, match="Evaluator returned mismatched result count"
        ):
            evaluator.evaluate([0, 1], {})

        class _NoPromotionEvaluator:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                return [
                    FitnessResult(objectives=(float(program),), metadata={})
                    for program in programs
                ]

        class _Passthrough:
            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                return [FitnessResult(objectives=(0.0,), metadata={}) for _ in programs]

        no_promote_evaluator = MultiFidelityFitnessEvaluator(
            levels=(
                _NoPromotionEvaluator(),
                _Passthrough(),
            ),
            top_k_per_level=1 / 2,
            objective_directions=("maximize",),
            promotion_strategy="direction_aware_first",
        )
        no_promote_evaluator._select_by_direction_aware_first = (
            lambda candidate_indices, _results, _budget: []
        )  # type: ignore[method-assign]
        no_promote_evaluator.evaluate([0, 1], {})

    def test_scalarized_weight_shortfall_and_crowding_branches(self) -> None:
        class _TwoObjectiveEvaluator:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                self.calls.append(tuple(programs))
                score_map = {
                    0: (1.0, 1.0),
                    1: (1.0, 1.0),
                    2: (1.0, 1.0),
                    3: (0.5, float("nan")),
                }
                return [
                    FitnessResult(
                        objectives=score_map[int(program)],
                        metadata={},
                    )
                    for program in programs
                ]

        level0 = _TwoObjectiveEvaluator()
        level1 = _ScoreByProgramEvaluator(
            [3.0, 2.0, 1.0, 0.0], objectives_per_program=1
        )
        evaluator = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=2 / 4,
            objective_directions=("maximize", "minimize"),
            promotion_strategy="scalarized",
            scalarization_weights=(1.0,),
        )
        evaluator.evaluate([0, 1, 2, 3], {})
        assert level1.calls == [(0, 1)]

    def test_pareto_minimize_and_niche_fallback_branches(self) -> None:
        class _MinimizeEvaluator:
            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                score_map = {
                    0: (1.0, 2.0),
                    1: (2.0, 1.0),
                    2: (3.0, 3.0),
                }
                return [
                    FitnessResult(
                        objectives=score_map[int(program)],
                        metadata={},
                    )
                    for program in programs
                ]

        level0 = _MinimizeEvaluator()
        level1 = _ScoreByProgramEvaluator([2.0, 1.5, 1.0], objectives_per_program=1)
        evaluator_minimize = MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=1,
            objective_directions=("minimize", "minimize"),
            promotion_strategy="pareto",
        )
        evaluator_minimize.evaluate([0, 1, 2], {})
        assert len(level1.calls) == 1

        class _NoMetadataEvaluator:
            def evaluate(
                self, programs: Sequence[int], context: dict[str, object]
            ) -> list[FitnessResult]:
                del context
                return [
                    FitnessResult(
                        objectives=(score,),
                        metadata={},
                    )
                    for score, _ in zip(
                        (3.0, 2.0, 1.0),
                        programs,
                        strict=False,
                    )
                ]

        no_meta_evaluator = MultiFidelityFitnessEvaluator(
            levels=(_NoMetadataEvaluator(), _ScoreByProgramEvaluator([5.0, 4.0, 3.0])),
            top_k_per_level=1 / 2,
            objective_directions=("maximize",),
            promotion_strategy="niche",
            fallback_niche_metadata_keys=("missing_key",),
        )
        assert no_meta_evaluator.evaluate([0, 1, 2], {})[0].objectives == (5.0,)

        crowding_sorter = MultiFidelityFitnessEvaluator(
            levels=(_ScoreByProgramEvaluator([1.0]),),
            top_k_per_level=1,
            objective_directions=("maximize",),
        )
        direct = crowding_sorter._sort_by_crowding(
            [0, 1],
            [
                FitnessResult(objectives=(1.0,), metadata={}),
                FitnessResult(objectives=(1.0,), metadata={}),
            ],
        )
        assert direct == [0, 1]

        direct = crowding_sorter._sort_by_crowding(
            [0, 1],
            [
                FitnessResult(objectives=(1.0,), metadata={}),
                FitnessResult(objectives=(float("nan"),), metadata={}),
            ],
        )
        assert direct == [0, 1]
