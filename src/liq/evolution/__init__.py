"""
liq-evolution: Trading strategy evolution layer for the LIQ Stack.

This package provides domain-specific integration between liq-gp (the GP
engine) and the trading strategy execution pipeline (liq-runner, liq-signals).
It handles primitive registration, multi-stage fitness evaluation, and
parallel program evaluation.
"""

from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
from liq.evolution.adapters.signal_output import GPSignalOutput
from liq.evolution.config import (
    EvolutionConfig,
    FitnessConfig,
    FitnessStageConfig,
    GPConfig,
    ParallelConfig,
    PrimitiveConfig,
    SerializationConfig,
    WarmStartConfig,
    build_gp_config,
)
from liq.evolution.errors import (
    AdapterError,
    ConfigurationError,
    EvaluationError,
    EvolutionError,
    FitnessError,
    FitnessEvaluationError,
    LiqEvolutionError,
    ParallelExecutionError,
    PrimitiveError,
    PrimitiveSetupError,
    SerializationError,
)
from liq.evolution.evolution.engine import evolve
from liq.evolution.fitness.label_metrics import LabelFitnessEvaluator
from liq.evolution.fitness.objectives import wire_objectives
from liq.evolution.fitness.runner_backtest import BacktestFitnessEvaluator
from liq.evolution.fitness.two_stage import TwoStageFitnessEvaluator
from liq.evolution.primitives.feature_context import FeatureContext
from liq.evolution.primitives.registry import build_trading_registry
from liq.evolution.primitives.series_sources import prepare_evaluation_context
from liq.evolution.program import (
    ConstantNode,
    EvaluationContext,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
    evaluate,
)
from liq.evolution.program.genome import Genome
from liq.evolution.program.serialize import deserialize_genome, serialize_genome
from liq.evolution.protocols import (
    FitnessEvaluator,
    FitnessStageEvaluator,
    GPStrategy,
    IndicatorBackend,
    PrimitiveRegistry,
    StoreBackend,
)

__all__ = [
    # Configuration
    "EvolutionConfig",
    "GPConfig",
    "FitnessConfig",
    "SerializationConfig",
    "PrimitiveConfig",
    "FitnessStageConfig",
    "ParallelConfig",
    "WarmStartConfig",
    # Protocols
    "FitnessEvaluator",
    "IndicatorBackend",
    "GPStrategy",
    "FitnessStageEvaluator",
    "PrimitiveRegistry",
    "StoreBackend",
    # Program / AST
    "Program",
    "TerminalNode",
    "ConstantNode",
    "FunctionNode",
    "ParameterizedNode",
    "EvaluationContext",
    "Genome",
    "evaluate",
    "serialize_genome",
    "deserialize_genome",
    # Builders
    "build_trading_registry",
    "prepare_evaluation_context",
    "build_gp_config",
    # Caching
    "FeatureContext",
    # Evolution engine
    "evolve",
    # Fitness
    "LabelFitnessEvaluator",
    "BacktestFitnessEvaluator",
    "TwoStageFitnessEvaluator",
    "wire_objectives",
    # Adapters
    "GPStrategyAdapter",
    "GPSignalOutput",
    # Errors
    "EvolutionError",
    "LiqEvolutionError",
    "PrimitiveSetupError",
    "FitnessEvaluationError",
    "AdapterError",
    "ConfigurationError",
    "ParallelExecutionError",
    "PrimitiveError",
    "EvaluationError",
    "SerializationError",
    "FitnessError",
]

__version__ = "0.1.0"
