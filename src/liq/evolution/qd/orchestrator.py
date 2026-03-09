"""Quality-diversity evolution orchestration helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from liq.evolution.fitness.evaluation_schema import (
    METADATA_KEY_BEHAVIOR_DESCRIPTORS,
    METADATA_KEY_RAW_OBJECTIVES,
)
from liq.gp.config import GPConfig
from liq.gp.evolution.engine import evolve
from liq.gp.evolution.qd_archive import QDArchive
from liq.gp.evolution.selection import select
from liq.gp.program.ast import Program
from liq.gp.types import EvolutionResult, FitnessResult, SelectionContext


@dataclass(frozen=True)
class QDEvolutionResult:
    """Structured output for QD-enabled evolution runs."""

    evolution_result: EvolutionResult
    archive: Any
    portfolio: list[Any]
    coverage_report: dict[str, Any]
    behavior_descriptor_names: tuple[str, ...]


def _safe_float(value: Any) -> float | None:
    """Coerce a value to float and filter non-finite values."""
    if not isinstance(value, (int, float, np.number)):
        return None
    parsed = float(value)
    if not np.isfinite(parsed):
        return None
    return parsed


def _extract_objectives(fitness: FitnessResult) -> tuple[float, ...] | None:
    raw = fitness.metadata.get(METADATA_KEY_RAW_OBJECTIVES, fitness.objectives)
    if not isinstance(raw, (tuple, list)):
        return None
    values: list[float] = []
    for raw_value in raw:
        parsed = _safe_float(raw_value)
        if parsed is None:
            return None
        values.append(parsed)
    return tuple(values) if values else None


def _extract_descriptors(
    fitness: FitnessResult,
    descriptor_names: tuple[str, ...],
) -> tuple[float, ...] | None:
    raw_descriptors = fitness.metadata.get(METADATA_KEY_BEHAVIOR_DESCRIPTORS)
    if not isinstance(raw_descriptors, Mapping):
        return None
    values: list[float] = []
    for name in descriptor_names:
        parsed = _safe_float(raw_descriptors.get(name))
        if parsed is None:
            return None
        if parsed < 0.0:
            parsed = 0.0
        if parsed > 1.0:
            parsed = 1.0
        values.append(parsed)
    return tuple(values)


def _seed_archive(
    archive: Any,
    population: list[Any],
    fitnesses: list[FitnessResult],
    descriptor_names: tuple[str, ...],
) -> None:
    for individual, fitness in zip(population, fitnesses, strict=False):
        objectives = _extract_objectives(fitness)
        descriptors = _extract_descriptors(fitness, descriptor_names)
        if objectives is None or descriptors is None:
            continue
        try:
            archive.insert(individual, objectives, descriptors)
        except Exception:
            # Archive insertion is a best-effort optimization path; bad
            # descriptors/objectives should not hard-fail evolution.
            continue


def _qd_parent_source(
    archive: Any,
    descriptor_names: tuple[str, ...],
    *,
    coverage_weight: float = 0.3,
    coverage_interval: int = 1,
) -> Callable[
    [list[Any], list[FitnessResult], GPConfig, np.random.Generator, int, SelectionContext],
    list[Any],
]:
    generation = 0

    def source(
        population: list[Any],
        fitnesses: list[FitnessResult],
        config: GPConfig,
        rng: np.random.Generator,
        target_size: int,
        sel_ctx: SelectionContext,
    ) -> list[Program]:
        nonlocal generation
        _seed_archive(archive, population, fitnesses, descriptor_names)

        use_coverage = coverage_interval > 0 and generation % coverage_interval == 0
        generation += 1

        selected: list[Program] = []
        remaining = target_size
        if (
            use_coverage
            and target_size > 0
            and archive.filled_bins > 0
            and coverage_weight > 0.0
        ):
            n_archive = int(target_size * coverage_weight)
            if n_archive == 0:
                n_archive = 1
            if n_archive > target_size:
                n_archive = target_size
            selected.extend(
                cast(
                    list[Program],
                    archive.sample(n_archive, rng, coverage_weight=coverage_weight),
                )
            )
            remaining = target_size - len(selected)

        if remaining > 0:
            selected.extend(
                select(
                    population,
                    fitnesses,
                    config,
                    rng,
                    fronts=sel_ctx.fronts,
                    ranks=sel_ctx.ranks,
                    crowding=sel_ctx.crowding,
                )[:remaining]
            )

        if len(selected) > target_size:
            selected = selected[:target_size]
        return selected

    return source


def run_qd_evolution(
    registry: Any,
    config: GPConfig,
    evaluator: Any,
    context: Mapping[str, Any],
    *,
    archive: Any | None = None,
    behavior_descriptor_names: Sequence[str] | None = None,
    archive_bins_per_dim: int | tuple[int, ...] = 4,
    archive_bin_capacity: int = 1,
    coverage_weight: float = 0.3,
    coverage_interval: int = 1,
    portfolio_size: int | None = None,
) -> QDEvolutionResult:
    """Run evolution with a QD archive and archive-aware parent sourcing.

    Args:
        registry: Primitive registry for GP construction.
        config: GP execution config.
        evaluator: Fitness evaluator with ``evaluate(programs, context)``.
        context: Evaluation context mapping.
        archive: Optional pre-created archive; if omitted, one is created.
        behavior_descriptor_names: Descriptor fields read from
            ``FitnessResult.metadata[METADATA_KEY_BEHAVIOR_DESCRIPTORS]``.
            Defaults to ``("turnover",)`` for compatibility.
        archive_bins_per_dim: Binning resolution passed to QDArchive.
        archive_bin_capacity: Per-bin capacity used by QDArchive.
        coverage_weight: Share of parents coming from archive sampling.
            ``0`` disables coverage pressure.
        coverage_interval: Frequency (in generations) for coverage-aware sampling.
        portfolio_size: Number of elites/portfolio entries to return.
    """
    if coverage_weight < 0.0 or coverage_weight > 1.0:
        raise ValueError("coverage_weight must be in [0.0, 1.0]")
    if coverage_interval < 1:
        raise ValueError("coverage_interval must be >= 1")
    if portfolio_size is not None and portfolio_size < 1:
        raise ValueError("portfolio_size must be >= 1 when provided")

    descriptor_names = tuple(
        behavior_descriptor_names
        if behavior_descriptor_names is not None
        else ("turnover",)
    )
    if not descriptor_names:
        raise ValueError("behavior_descriptor_names must be non-empty")

    if archive is None:
        archive = QDArchive(
            n_dims=len(descriptor_names),
            bins_per_dim=archive_bins_per_dim,
            descriptor_bounds=((0.0, 1.0),) * len(descriptor_names),
            objective_directions=list(config.fitness.objective_directions),
            bin_capacity=archive_bin_capacity,
        )

    if len(descriptor_names) != archive.n_dims:
        raise ValueError(
            f"descriptor_names length ({len(descriptor_names)}) must equal "
            f"archive.n_dims ({archive.n_dims})"
        )

    source = _qd_parent_source(
        archive=archive,
        descriptor_names=descriptor_names,
        coverage_weight=coverage_weight,
        coverage_interval=coverage_interval,
    )

    evolution_result = evolve(
        registry=registry,
        config=config,
        evaluator=evaluator,
        context=dict(context),
        parent_source=source,
    )

    if portfolio_size is None:
        portfolio: list[Any] = cast(list[Any], archive.elites())
    else:
        portfolio = cast(list[Any], archive.elites())[:portfolio_size]
    if not portfolio:
        portfolio = evolution_result.pareto_front[:portfolio_size]

    return QDEvolutionResult(
        evolution_result=evolution_result,
        archive=archive,
        portfolio=portfolio,
        coverage_report=archive.coverage_report(),
        behavior_descriptor_names=descriptor_names,
    )
