"""
liq-evolution: Trading strategy evolution layer for the LIQ Stack.

This package provides domain-specific integration between liq-gp (the GP
engine) and the trading strategy execution pipeline (liq-runner, liq-signals).
It handles primitive registration, multi-stage fitness evaluation, and
parallel program evaluation.
"""

from liq.evolution.adapters.artifact_store import LiqStoreEvolutionArtifactStore
from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
from liq.evolution.adapters.signal_output import GPSignalOutput
from liq.evolution.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    KNOWN_REJECTION_REASON_CODES,
    DependencyFingerprint,
    EvolutionRunArtifact,
    RejectionEvent,
    is_known_rejection_reason,
    require_known_rejection_reason,
)
from liq.evolution.config import (
    EvolutionConfig,
    EvolutionRunConfig,
    FitnessConfig,
    FitnessStageConfig,
    GPConfig,
    ParallelConfig,
    PrimitiveConfig,
    RegimeGateConfig,
    SerializationConfig,
    WarmStartConfig,
    build_gp_config,
)
from liq.evolution.errors import (
    AdapterError,
    CandidateArtifactError,
    ConfigurationError,
    DeterminismViolationError,
    EvaluationContractError,
    EvaluationError,
    EvolutionError,
    FitnessError,
    FitnessEvaluationError,
    LiqEvolutionError,
    ParallelExecutionError,
    PrimitiveError,
    PrimitiveSetupError,
    ProtocolVersionError,
    SerializationError,
    StrategyArtifactError,
)
from liq.evolution.evolution.engine import evolve
from liq.evolution.fitness.eval_cache import FitnessEvaluationCache  # noqa: F401
from liq.evolution.fitness.label_metrics import LabelFitnessEvaluator
from liq.evolution.fitness.multifidelity import MultiFidelityFitnessEvaluator
from liq.evolution.fitness.objectives import wire_objectives
from liq.evolution.fitness.runner_backtest import BacktestFitnessEvaluator
from liq.evolution.fitness.strategy_evaluator import StrategyEvaluator
from liq.evolution.fitness.two_stage import TwoStageFitnessEvaluator
from liq.evolution.presets import (
    RegimeOperationalPreset,
    get_regime_preset,
    list_regime_presets,
)
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
    EVOLUTION_PROTOCOL_VERSION,
    GP_PROTOCOL_VERSION,
    CandidateArtifact,
    CandidateEvaluator,
    EvolutionArtifactStore,
    FitnessEvaluator,
    FitnessStageEvaluator,
    GPStrategy,
    IndicatorBackend,
    PrimitiveRegistry,
    StoreBackend,
    StrategyArtifact,
)
from liq.evolution.qd import QDEvolutionResult, run_qd_evolution
from liq.evolution.regime_model import (
    RegimeDetector,
    RegimeExpert,
    RegimeGate,
    RegimeId,
    RegimeModel,
    RegimeRisk,
    RegimeWeights,
)
from liq.evolution.seeds import (
    SeedInjectionCadence,
    SeedManifest,
    SeedSpec,
    SeedTemplateRole,
    StrategySeedTemplate,
    build_seed_champion_payload,
    build_seed_from_spec,
    build_seed_programs,
    build_seed_programs_from_path,
    build_strategy_seed,
    build_strategy_seeds,
    built_in_seed_payloads,
    get_seed_template,
    inject_seed_programs_from_champion_pool,
    list_known_strategy_seeds,
    list_seed_templates_by_role,
    load_seed_manifest,
    load_seed_specs,
    program_signature,
    select_seed_champion_pool,
    should_inject,
    validate_external_seed_payload,
)
from liq.gp.config import SeedInjectionConfig

__all__ = [
    # Configuration
    "EvolutionConfig",
    "EvolutionRunConfig",
    "RegimeGateConfig",
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
    "LiqStoreEvolutionArtifactStore",
    "CandidateArtifact",
    "StrategyArtifact",
    "CandidateEvaluator",
    "EvolutionArtifactStore",
    "GP_PROTOCOL_VERSION",
    "EVOLUTION_PROTOCOL_VERSION",
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
    "SeedInjectionConfig",
    "SeedInjectionCadence",
    "inject_seed_programs_from_champion_pool",
    "select_seed_champion_pool",
    "should_inject",
    "program_signature",
    "build_seed_champion_payload",
    "validate_external_seed_payload",
    "built_in_seed_payloads",
    "list_seed_templates_by_role",
    "build_strategy_seed",
    "build_strategy_seeds",
    "build_seed_from_spec",
    "build_seed_programs",
    "build_seed_programs_from_path",
    "load_seed_manifest",
    "load_seed_specs",
    "SeedManifest",
    "SeedSpec",
    "get_seed_template",
    "list_known_strategy_seeds",
    "SeedTemplateRole",
    "StrategySeedTemplate",
    "RegimeId",
    "RegimeWeights",
    "RegimeDetector",
    "RegimeGate",
    "RegimeExpert",
    "RegimeRisk",
    "RegimeModel",
    "EvolutionRunArtifact",
    "DependencyFingerprint",
    "RejectionEvent",
    "KNOWN_REJECTION_REASON_CODES",
    "ARTIFACT_SCHEMA_VERSION",
    "is_known_rejection_reason",
    "require_known_rejection_reason",
    # Caching
    "FeatureContext",
    # Evolution engine
    "evolve",
    # Fitness
    "LabelFitnessEvaluator",
    "BacktestFitnessEvaluator",
    "FitnessEvaluationCache",
    "MultiFidelityFitnessEvaluator",
    "TwoStageFitnessEvaluator",
    "StrategyEvaluator",
    "wire_objectives",
    "run_qd_evolution",
    "QDEvolutionResult",
    # Presets
    "RegimeOperationalPreset",
    "list_regime_presets",
    "get_regime_preset",
    # Adapters
    "GPStrategyAdapter",
    "GPSignalOutput",
    # Errors
    "EvolutionError",
    "CandidateArtifactError",
    "StrategyArtifactError",
    "LiqEvolutionError",
    "EvaluationContractError",
    "PrimitiveSetupError",
    "FitnessEvaluationError",
    "AdapterError",
    "DeterminismViolationError",
    "ProtocolVersionError",
    "ConfigurationError",
    "ParallelExecutionError",
    "PrimitiveError",
    "EvaluationError",
    "SerializationError",
    "FitnessError",
]
