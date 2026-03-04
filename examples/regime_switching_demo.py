#!/usr/bin/env python3
"""Demonstrate regime-state transitions and fallback reasons."""

from __future__ import annotations

import numpy as np
import polars as pl

from liq.evolution import (
    GPStrategyAdapter,
    PrimitiveConfig,
    build_gp_config,
    build_trading_registry,
)
from liq.evolution.config import EvolutionConfig
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import FunctionNode, TerminalNode
from liq.gp.types import Series


def build_regime_program(registry: PrimitiveRegistry) -> FunctionNode:
    sub = registry.get("sub")
    return FunctionNode(
        sub,
        (
            TerminalNode(name="close", output_type=Series),
            TerminalNode(name="open", output_type=Series),
        ),
    )


def make_price_path() -> tuple[np.ndarray, np.ndarray]:
    open_ = np.array([101.0, 101.0, 102.0, 101.0, 100.0, 102.0, 103.0, 102.0], dtype=float)
    close = np.array([102.0, 100.0, 104.0, 99.0, 99.5, 105.0, 101.0, 103.0], dtype=float)
    return open_, close


def make_transition_price_path() -> tuple[np.ndarray, np.ndarray]:
    open_ = np.array([100.0, 101.0, 102.0, 102.5, 103.0, 103.2, 102.8, 103.0, 104.0], dtype=float)
    close = np.array([101.0, 101.5, 101.0, 101.2, 102.0, 101.2, 100.8, 100.4, 101.8], dtype=float)
    return open_, close


def build_adapter() -> GPStrategyAdapter:
    registry = build_trading_registry(PrimitiveConfig())
    program = build_regime_program(registry)
    gp_config = build_gp_config(
        EvolutionConfig(
            population_size=24,
            max_depth=3,
            generations=1,
            run={"stage_b_candidate_budget": 4},
        )
    )
    adapter = GPStrategyAdapter(
        registry,
        gp_config,
        evaluator=None,
        regime_confidence_threshold=0.55,
        regime_occupancy_threshold=0.10,
        regime_hysteresis_margin=0.05,
        regime_min_persistence=2,
    )
    adapter._program = program
    return adapter


def to_context(
    open_: np.ndarray,
    close: np.ndarray,
    *,
    confidence: float | np.ndarray = 0.95,
    occupancy: float | np.ndarray = 1.0,
) -> pl.DataFrame:
    data = {
        "open": open_,
        "close": close,
        "high": close + 0.6,
        "low": close - 0.6,
        "volume": np.linspace(1_000_000.0, 1_200_000.0, open_.size),
    }
    if isinstance(confidence, np.ndarray):
        data["regime_confidence"] = confidence
    else:
        data["regime_confidence"] = np.full(open_.size, float(confidence), dtype=float)
    if isinstance(occupancy, np.ndarray):
        data["regime_occupancy"] = occupancy
    else:
        data["regime_occupancy"] = np.full(open_.size, float(occupancy), dtype=float)
    return pl.DataFrame(data)


def print_output(header: str, out: object) -> None:
    state = out.regime_state
    scores = out.scores.to_list()
    print(f"{header: <28} label={state.label} reason={state.reason_code}")
    print(f"  scores={scores}")
    print(f"  metadata={out.metadata}")


def run_demo() -> None:
    adapter = build_adapter()

    open_, close_ = make_price_path()
    base_context = to_context(open_, close_)
    print_output("high-confidence regime", adapter.predict(base_context))

    low_conf = to_context(
        open_,
        close_,
        confidence=np.full(close_.size, 0.20, dtype=float),
    )
    print_output("low-confidence fallback", adapter.predict(low_conf))

    missing_features = pl.DataFrame(
        {
            "open": open_,
            "close": close_,
            "high": close_ + 0.6,
            "low": close_ - 0.6,
            "volume": np.linspace(1_000_000.0, 1_200_000.0, close_.size),
        }
    )
    print_output("missing regime features", adapter.predict(missing_features))

    open_transition, close_transition = make_transition_price_path()
    transition_context = to_context(
        open_transition,
        close_transition,
        confidence=0.98,
        occupancy=np.linspace(0.5, 1.0, close_transition.size),
    )
    print_output("transition path (hysteresis)", adapter.predict(transition_context))


if __name__ == "__main__":
    run_demo()
