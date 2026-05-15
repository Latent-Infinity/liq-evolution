"""Phase 4 paired detector-source evolution evidence runner."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from liq.core import RegimeId
from liq.gp.config import GPConfig
from liq.gp.evolution.engine import evolve
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import Program
from liq.gp.program.eval import evaluate
from liq.gp.types import BoolSeries, FitnessResult, GenerationStats, Series

DetectorSource = Literal["evolved", "trained"]
REQUIRED_TRAINED_TERMINALS: tuple[str, ...] = tuple(
    f"svm_regime_is_{regime.value}"
    for regime in (
        RegimeId.trend,
        RegimeId.range,
        RegimeId.neutral,
        RegimeId.fallback,
        RegimeId.no_trade,
        RegimeId.empty,
    )
)
_PARITY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Phase4Evidence:
    """Trusted trained detector materialization used for Phase 4 evolution."""

    terminal_frame: Mapping[str, np.ndarray]
    terminal_names: tuple[str, ...]
    handoff_classifier_path: str | None
    handoff_classifier_sha256: str | None
    handoff_parity_match_rate: float
    row_count: int


@dataclass(frozen=True)
class Phase4RunSummary:
    """Serializable summary for one detector-source evolution run."""

    seed: int
    detector_source: DetectorSource
    best_fitness: float
    best_program_size: int
    final_mean_fitness: float
    time_to_final_best_generation: int | None
    fitness_curve: list[dict[str, float | int]]
    trained_terminal_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "detector_source": self.detector_source,
            "best_fitness": self.best_fitness,
            "best_program_size": self.best_program_size,
            "final_mean_fitness": self.final_mean_fitness,
            "time_to_final_best_generation": self.time_to_final_best_generation,
            "fitness_curve": self.fitness_curve,
            "trained_terminal_names": list(self.trained_terminal_names),
        }


class DetectorFitnessEvaluator:
    """Evaluate detector programs by matching a BTC-derived target series."""

    def __init__(self, target: np.ndarray) -> None:
        self._target = target.astype(float)

    def evaluate(
        self, programs: list[Program], context: dict[str, np.ndarray]
    ) -> list[FitnessResult]:
        results: list[FitnessResult] = []
        for program in programs:
            try:
                output = np.asarray(evaluate(program, context), dtype=float)
                mse = float(np.mean((output - self._target) ** 2))
                results.append(FitnessResult(objectives=(-mse,), metadata={"mse": mse}))
            except Exception:
                results.append(
                    FitnessResult(objectives=(-1e10,), metadata={"mse": 1e10})
                )
        return results


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_bool_series(
    name: str, values: object, *, expected_length: int | None
) -> np.ndarray:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise ValueError(f"trained terminal {name!r} must be a sequence")
    series = np.asarray([bool(value) for value in values], dtype=float)
    if series.size == 0:
        raise ValueError(f"trained terminal {name!r} cannot be empty")
    if expected_length is not None and series.size != expected_length:
        raise ValueError("all trained terminal series must have the same length")
    return series


def _load_evidence(
    path: Path, *, trusted_classifier_path: Path | None = None
) -> Phase4Evidence:
    data = json.loads(path.read_text())
    terminal_frame = data.get("trained_terminal_frame")
    if not isinstance(terminal_frame, dict):
        raise ValueError(f"Phase 4 evidence missing trained_terminal_frame: {path}")

    terminal_names = data.get("trained_terminal_names")
    if terminal_names is None:
        terminal_names = list(terminal_frame)
    if not isinstance(terminal_names, list) or not all(
        isinstance(name, str) for name in terminal_names
    ):
        raise ValueError(
            "Phase 4 evidence trained_terminal_names must be a list of strings"
        )

    missing = [
        name for name in REQUIRED_TRAINED_TERMINALS if name not in terminal_frame
    ]
    if missing:
        raise ValueError(
            f"Phase 4 evidence missing trained terminal columns: {missing}"
        )
    if tuple(terminal_names) != REQUIRED_TRAINED_TERMINALS:
        raise ValueError(
            "Phase 4 evidence trained_terminal_names must list all six required terminals in canonical order"
        )

    expected_length: int | None = None
    coerced: dict[str, np.ndarray] = {}
    for name in REQUIRED_TRAINED_TERMINALS:
        series = _coerce_bool_series(
            name, terminal_frame[name], expected_length=expected_length
        )
        expected_length = series.size
        coerced[name] = series

    parity = data.get("handoff_parity_match_rate")
    if not isinstance(parity, int | float) or not math.isfinite(float(parity)):
        raise ValueError("Phase 4 evidence missing finite handoff_parity_match_rate")
    parity_rate = float(parity)
    if not math.isclose(parity_rate, 1.0, abs_tol=_PARITY_TOLERANCE):
        raise ValueError(
            f"Phase 4 handoff parity gate failed: {parity_rate:.12f} != 1.0"
        )

    evidence_classifier_path = data.get("handoff_classifier_path")
    if evidence_classifier_path is not None and not isinstance(
        evidence_classifier_path, str
    ):
        raise ValueError(
            "Phase 4 evidence handoff_classifier_path must be a string when present"
        )

    evidence_sha256 = data.get("handoff_classifier_sha256")
    if evidence_sha256 is not None and not isinstance(evidence_sha256, str):
        raise ValueError(
            "Phase 4 evidence handoff_classifier_sha256 must be a string when present"
        )
    if trusted_classifier_path is not None:
        if not trusted_classifier_path.exists():
            raise ValueError(
                f"trusted classifier artifact does not exist: {trusted_classifier_path}"
            )
        trusted_sha256 = _sha256_file(trusted_classifier_path)
        if evidence_sha256 is None:
            raise ValueError("Phase 4 evidence missing handoff_classifier_sha256")
        if trusted_sha256 != evidence_sha256:
            raise ValueError(
                "trusted classifier artifact hash does not match Phase 4 evidence"
            )

    return Phase4Evidence(
        terminal_frame=coerced,
        terminal_names=REQUIRED_TRAINED_TERMINALS,
        handoff_classifier_path=evidence_classifier_path,
        handoff_classifier_sha256=evidence_sha256,
        handoff_parity_match_rate=parity_rate,
        row_count=expected_length or 0,
    )


def _build_context(
    evidence: Phase4Evidence, *, detector_source: DetectorSource, seed: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    trend = evidence.terminal_frame["svm_regime_is_trend"]
    range_ = evidence.terminal_frame["svm_regime_is_range"]
    neutral = evidence.terminal_frame["svm_regime_is_neutral"]
    target = np.clip(trend + (0.5 * range_) + (0.25 * neutral), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    baseline = rng.uniform(0.0, 1.0, size=evidence.row_count)
    context = {"baseline_signal": baseline}
    if detector_source == "trained":
        context.update(
            {
                name: np.asarray(values, dtype=float)
                for name, values in evidence.terminal_frame.items()
            }
        )
    else:
        proxy = np.roll(target, seed % max(1, evidence.row_count))
        noise_mask = rng.uniform(0.0, 1.0, size=evidence.row_count) < 0.15
        context["evolved_detector_proxy"] = np.where(noise_mask, 1.0 - proxy, proxy)
    return context, target


def _registry(detector_source: DetectorSource) -> PrimitiveRegistry:
    registry = PrimitiveRegistry()
    registry.register(
        "baseline_signal",
        lambda: None,
        category="terminal",
        input_types=(),
        output_type=Series,
    )
    if detector_source == "trained":
        for name in REQUIRED_TRAINED_TERMINALS:
            registry.register(
                name,
                lambda: None,
                category="trained_detector",
                input_types=(),
                output_type=BoolSeries,
            )
        registry.register(
            "and_op",
            lambda left, right: np.where((left > 0.5) & (right > 0.5), 1.0, 0.0),
            category="logic",
            input_types=(BoolSeries, BoolSeries),
            output_type=BoolSeries,
        )
        registry.register(
            "or_op",
            lambda left, right: np.where((left > 0.5) | (right > 0.5), 1.0, 0.0),
            category="logic",
            input_types=(BoolSeries, BoolSeries),
            output_type=BoolSeries,
        )
        registry.register(
            "not_op",
            lambda value: np.where(value > 0.5, 0.0, 1.0),
            category="logic",
            input_types=(BoolSeries,),
            output_type=BoolSeries,
        )
        registry.register(
            "if_then_else",
            lambda cond, left, right: np.where(cond > 0.5, left, right),
            category="logic",
            input_types=(BoolSeries, Series, Series),
            output_type=Series,
        )
    else:
        registry.register(
            "evolved_detector_proxy",
            lambda: None,
            category="evolved_detector",
            input_types=(),
            output_type=Series,
        )
    registry.register(
        "add",
        lambda left, right: left + right,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    registry.register(
        "sub",
        lambda left, right: left - right,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    registry.register(
        "mul",
        lambda left, right: left * right,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    registry.register(
        "avg",
        lambda left, right: (left + right) / 2.0,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    return registry


def _time_to_threshold(
    curve: list[dict[str, float | int]], threshold: float
) -> int | None:
    for item in curve:
        if float(item["best_fitness"]) >= threshold:
            return int(item["generation"])
    return None


def _curve(history: list[GenerationStats]) -> list[dict[str, float | int]]:
    return [
        {
            "generation": item.generation,
            "best_fitness": item.best_fitness[0],
            "mean_fitness": item.mean_fitness[0],
            "best_program_size": item.best_program_size,
            "mean_program_size": item.mean_program_size,
        }
        for item in history
    ]


def run_detector_source(
    *,
    seed: int,
    detector_source: DetectorSource,
    evidence_path: Path,
    output_dir: Path,
    population_size: int = 30,
    generations: int = 8,
    trusted_classifier_path: Path | None = None,
) -> Phase4RunSummary:
    evidence = _load_evidence(
        evidence_path, trusted_classifier_path=trusted_classifier_path
    )
    context, target = _build_context(
        evidence, detector_source=detector_source, seed=seed
    )
    result = evolve(
        _registry(detector_source),
        GPConfig(
            population_size=population_size,
            generations=generations,
            max_depth=4,
            max_size=31,
            seed=seed,
            elitism_count=2,
            tournament_size=3,
        ),
        DetectorFitnessEvaluator(target),
        context,
    )
    curve = _curve(result.fitness_history)
    best_fitness = float(curve[-1]["best_fitness"]) if curve else float("nan")
    summary = Phase4RunSummary(
        seed=seed,
        detector_source=detector_source,
        best_fitness=best_fitness,
        best_program_size=int(curve[-1]["best_program_size"]) if curve else 0,
        final_mean_fitness=float(curve[-1]["mean_fitness"]) if curve else float("nan"),
        time_to_final_best_generation=_time_to_threshold(curve, best_fitness),
        fitness_curve=curve,
        trained_terminal_names=evidence.terminal_names,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evolution_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True)
    )
    return summary


def run_paired_detector_swap(
    *,
    seed: int,
    evidence_path: Path,
    output_root: Path,
    population_size: int = 30,
    generations: int = 8,
    trusted_classifier_path: Path | None = None,
) -> dict[str, Any]:
    evidence = _load_evidence(
        evidence_path, trusted_classifier_path=trusted_classifier_path
    )
    evolved = run_detector_source(
        seed=seed,
        detector_source="evolved",
        evidence_path=evidence_path,
        output_dir=output_root / "evolved" / f"seed-{seed}",
        population_size=population_size,
        generations=generations,
        trusted_classifier_path=trusted_classifier_path,
    )
    trained = run_detector_source(
        seed=seed,
        detector_source="trained",
        evidence_path=evidence_path,
        output_dir=output_root / "trained" / f"seed-{seed}",
        population_size=population_size,
        generations=generations,
        trusted_classifier_path=trusted_classifier_path,
    )
    comparison = {
        "seed": seed,
        "population_size": population_size,
        "generations": generations,
        "trusted_classifier_path": str(trusted_classifier_path)
        if trusted_classifier_path is not None
        else evidence.handoff_classifier_path,
        "handoff_classifier_sha256": evidence.handoff_classifier_sha256,
        "handoff_parity_match_rate": evidence.handoff_parity_match_rate,
        "trained_terminal_names": list(evidence.terminal_names),
        "evolved": evolved.to_dict(),
        "trained": trained.to_dict(),
        "best_fitness_delta_trained_minus_evolved": trained.best_fitness
        - evolved.best_fitness,
        "final_mean_fitness_delta_trained_minus_evolved": trained.final_mean_fitness
        - evolved.final_mean_fitness,
    }
    seed_dir = output_root / "comparison" / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "paired_evolution.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True)
    )
    return comparison
