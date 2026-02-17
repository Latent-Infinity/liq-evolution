"""Tests for liq-gp re-exports through liq-evolution."""

from __future__ import annotations


class TestEvolutionReexports:
    def test_evolve_importable(self) -> None:
        from liq.evolution.evolution import evolve

        assert callable(evolve)

    def test_population_init_importable(self) -> None:
        from liq.evolution.evolution import (
            generate_full,
            generate_grow,
            initialize_population,
            validate_seed_programs,
        )

        assert callable(generate_full)
        assert callable(generate_grow)
        assert callable(initialize_population)
        assert callable(validate_seed_programs)

    def test_operators_importable(self) -> None:
        from liq.evolution.evolution import (
            hoist_mutation,
            parameter_mutation,
            point_mutation,
            subtree_crossover,
            subtree_mutation,
        )

        assert callable(subtree_crossover)
        assert callable(subtree_mutation)
        assert callable(point_mutation)
        assert callable(parameter_mutation)
        assert callable(hoist_mutation)

    def test_selection_importable(self) -> None:
        from liq.evolution.evolution import (
            crowding_distance,
            get_elites,
            non_dominated_sort,
            nsga2_select,
            select,
            tournament_select,
        )

        assert callable(tournament_select)
        assert callable(nsga2_select)
        assert callable(select)
        assert callable(get_elites)
        assert callable(non_dominated_sort)
        assert callable(crowding_distance)

    def test_constraints_importable(self) -> None:
        from liq.evolution.evolution import (
            apply_parsimony,
            enforce_constraints,
            filter_population,
        )

        assert callable(enforce_constraints)
        assert callable(apply_parsimony)
        assert callable(filter_population)

    def test_diversity_importable(self) -> None:
        from liq.evolution.evolution import (
            compute_fingerprint,
            deduplicate_population,
            sample_reference_context,
        )

        assert callable(compute_fingerprint)
        assert callable(deduplicate_population)
        assert callable(sample_reference_context)


class TestProgramReexports:
    def test_serialize_importable(self) -> None:
        from liq.evolution.program.serialize import (
            deserialize,
            deserialize_result,
            serialize,
            serialize_result,
        )

        assert callable(serialize)
        assert callable(deserialize)
        assert callable(serialize_result)
        assert callable(deserialize_result)

    def test_simplify_importable(self) -> None:
        from liq.evolution.program.simplify import (
            SimplificationRegistry,
            default_rules,
            simplify,
        )

        assert callable(simplify)
        assert callable(default_rules)
        assert SimplificationRegistry is not None

    def test_constants_importable(self) -> None:
        from liq.evolution.program.constants import (
            extract_constants,
            inject_constants,
            optimize_constants,
            select_for_optimization,
        )

        assert callable(extract_constants)
        assert callable(inject_constants)
        assert callable(optimize_constants)
        assert callable(select_for_optimization)
