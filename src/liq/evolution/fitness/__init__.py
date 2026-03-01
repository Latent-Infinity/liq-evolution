"""Fitness evaluation stages and contract helpers for trading strategy evolution."""

from liq.evolution.fitness.evaluation_schema import (  # noqa: F401
    BEHAVIOR_DESCRIPTOR_BENCHMARK_CORRELATION,
    BEHAVIOR_DESCRIPTOR_DRAWDOWN_PROFILE,
    BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
    BEHAVIOR_DESCRIPTOR_MAX_LEVERAGE,
    BEHAVIOR_DESCRIPTOR_NET_EXPOSURE,
    BEHAVIOR_DESCRIPTOR_TRADE_FREQUENCY,
    BEHAVIOR_DESCRIPTOR_TURNOVER,
    METADATA_KEY_BEHAVIOR_DESCRIPTORS,
    METADATA_KEY_CONSTRAINT_VIOLATIONS,
    METADATA_KEY_PER_SPLIT_METRICS,
    METADATA_KEY_RAW_OBJECTIVES,
    METADATA_KEY_SLICE_SCORES,
    ObjectiveDirection,
    SchemaValidationError,
    SLICE_TYPE_EVENT,
    SLICE_TYPE_INSTRUMENT,
    SLICE_TYPE_LIQUIDITY,
    SLICE_TYPE_TIME_WINDOW,
    SUPPORTED_BEHAVIOR_DESCRIPTORS,
    validate_evaluation_metadata,
    validate_objective_vector,
    validate_slice_id,
    to_loss_form,
)
from liq.evolution.fitness.label_metrics import LabelFitnessEvaluator  # noqa: F401
from liq.evolution.fitness.eval_cache import (  # noqa: F401
    FitnessEvaluationCache,
    compute_fingerprint,
    compute_program_hash,
)
from liq.evolution.fitness.multifidelity import MultiFidelityFitnessEvaluator  # noqa: F401
from liq.evolution.fitness.objectives import wire_objectives  # noqa: F401
from liq.evolution.fitness.runner_backtest import BacktestFitnessEvaluator  # noqa: F401
from liq.evolution.fitness.strategy_evaluator import StrategyEvaluator  # noqa: F401
from liq.evolution.fitness.two_stage import TwoStageFitnessEvaluator  # noqa: F401
