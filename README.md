# liq-evolution

Trading strategy evolution layer for the LIQ Stack. Connects the domain-agnostic
[liq-gp](../liq-gp) genetic programming engine to trading-specific primitives,
fitness evaluation, and strategy serialization.

## Installation

```bash
uv add liq-evolution
```

Indicator primitives are available by default through
[liq-features](../liq-features), which uses [liq-ta](../liq-ta) as its
calculation backend.

## Quickstart

```python
import numpy as np

from liq.evolution import (
    EvolutionConfig,
    LabelFitnessEvaluator,
    build_gp_config,
    build_trading_registry,
    evolve,
    prepare_evaluation_context,
)
from liq.evolution.config import PrimitiveConfig

# 1. Build the trading primitive registry
registry = build_trading_registry(PrimitiveConfig())

# 2. Prepare synthetic OHLCV data
rng = np.random.default_rng(42)
n = 500
ohlcv = {
    "open": rng.normal(100, 5, n),
    "high": rng.normal(105, 5, n),
    "low": rng.normal(95, 5, n),
    "close": rng.normal(100, 5, n),
    "volume": rng.uniform(1e6, 1e7, n),
}

# 3. Build evaluation context (adds derived series like log_returns)
context = prepare_evaluation_context(ohlcv)

# Add labels: 1.0 where next close > current close
labels = np.zeros(n)
labels[:-1] = (ohlcv["close"][1:] > ohlcv["close"][:-1]).astype(float)
context["labels"] = labels

# 4. Configure evolution
evo_config = EvolutionConfig(
    population_size=100,
    generations=10,
    max_depth=6,
    seed=42,
)
gp_config = build_gp_config(evo_config)

# 5. Evolve
evaluator = LabelFitnessEvaluator(metric="f1")
result = evolve(
    registry=registry,
    config=gp_config,
    evaluator=evaluator,
    context=context,
)

best_fitness = result.fitness_history[-1].best_fitness[0]
print(f"Best fitness: {best_fitness:.4f}")
print(f"Best program: {result.best_program}")
```

## Configuration

### EvolutionConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `population_size` | `int` | 300 | Number of programs in the population (>= 10) |
| `max_depth` | `int` | 8 | Maximum GP program tree depth (>= 2) |
| `generations` | `int` | 50 | Number of evolution generations (>= 1) |
| `seed` | `int` | 42 | Random seed for reproducibility |
| `batch_size` | `int \| None` | `None` | Mini-batch size (`None` = full evaluation) |
| `full_eval_interval` | `int` | 10 | Full evaluation every N generations |
| `primitives` | `PrimitiveConfig` | all enabled | Which primitive categories to register |
| `fitness_stages` | `FitnessStageConfig` | label-only | Fitness evaluation pipeline |
| `parallel` | `ParallelConfig` | sequential | Parallel evaluation settings |
| `gp` | `GPConfig` | defaults | Core GP parameters (rates, tournament, elitism) |
| `warm_start` | `WarmStartConfig` | disabled | Seed programs for warm-starting |

### GPConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mutation_rate` | `float` | 0.2 | Overall mutation probability |
| `crossover_rate` | `float` | 0.8 | Crossover probability |
| `tournament_size` | `int` | 3 | Tournament selection size |
| `elitism_count` | `int` | 1 | Elite programs preserved each generation |

### PrimitiveConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_numeric_ops` | `bool` | `True` | Arithmetic operators (+, -, *, /) |
| `enable_comparison_ops` | `bool` | `True` | Comparison operators (>, <, ==) |
| `enable_logic_ops` | `bool` | `True` | Logical operators (and, or, not) |
| `enable_crossover_ops` | `bool` | `True` | Crossover detection operators |
| `enable_temporal_ops` | `bool` | `True` | Temporal/lag operators |
| `enable_series_sources` | `bool` | `True` | Price/volume series terminals |

## Architecture

```
liq-gp (engine)  -->  liq-evolution (domain)  -->  liq-runner (execution)
                          |
                          +-- primitives/     Trading operators + indicator wrappers
                          |      |
                          |      +-- liq-features  (indicator metadata + param grids)
                          |              |
                          |              +-- liq-ta (indicator computation backend)
                          +-- fitness/        Label metrics & backtest fitness
                          +-- program/        AST, Genome, serialization
                          +-- adapters/       Runner strategy, signal provider
                          +-- evolution/      Re-exports from liq-gp engine
                          +-- config.py       Pydantic v2 configuration
```

**liq-gp** provides the generic GP engine (population init, selection,
crossover, mutation, simplification). **liq-evolution** adds trading-domain
primitives (price series, crossover detection, and indicator primitives
registered through `liq-features`), multi-stage fitness evaluation
(label-based + backtest), and adapters for the execution layer.

## Export / Import

Serialize a winning strategy for deployment:

```python
from liq.evolution import Genome, serialize_genome, deserialize_genome

# Wrap the best program in a Genome
genome = Genome(entry_program=result.best_program)

# Serialize to a portable dict
payload = serialize_genome(genome)

# Deserialize back (requires the same registry)
restored = deserialize_genome(payload, registry)
assert restored.entry_program == genome.entry_program
```

## Exception Hierarchy

```
LiqEvolutionError
├── EvolutionError
│   ├── PrimitiveError
│   │   └── PrimitiveSetupError
│   ├── EvaluationError
│   ├── SerializationError
│   ├── FitnessError
│   │   └── FitnessEvaluationError
│   ├── AdapterError
│   ├── ConfigurationError
│   └── ParallelExecutionError
```

Catch `LiqEvolutionError` for a blanket handler, or individual subclasses
for fine-grained control. All invalid configurations raise
`ConfigurationError` at construction time (fail-fast).

## Development

```bash
uv run pytest                          # run tests
uv run ruff check src/ tests/          # lint
uv run ruff format src/ tests/         # format
uv run ty check src/                   # type check
```

## Operational Notes (Stage 6 hardening)

- **Walk-forward split failures**
  - Missing or malformed split metadata in `WalkForwardSplit` is validated before
    evaluation; invalid folds are rejected before market simulation starts.
  - `FoldResult.slice_id` is propagated to evaluator metadata and used as the
    cache key to keep walk-forward reuse behavior stable.
- **Evaluator contract faults**
  - Lexicase selection (`selection_mode=lexicase`) requires
    `METADATA_KEY_SLICE_SCORES` and a parseable `METADATA_KEY_RAW_OBJECTIVES`
    tuple. A mismatch raises `ValueError` early.
  - Constraint violations appear in `METADATA_KEY_CONSTRAINT_VIOLATIONS` and are
    added as additional slice-score penalty cases.
- **Evaluation cache behavior**
  - Cache key is `(strategy_hash, slice_id, evaluator_fingerprint)`.
  - Config hashes include objective names/directions, split behavior, cost-model
    options, and descriptor schema version.
  - Cache misses are expected when objective sets or schema versions change; this
    prevents stale reuse across experiment variants.
- **Multi-fidelity control**
  - Expensive levels should only evaluate promoted candidates. If promotion
    thresholds or strategy budgeting are too aggressive/lenient, coverage and
    sample quality degrade. Verify `top_k_per_level`, promotion strategy, and
    level ordering for your domain.

## Stage 10 examples

Runnable demonstration scripts:

```bash
python liq-evolution/examples/two_stage_evolution.py
python liq-evolution/examples/regime_switching_demo.py
python liq-evolution/examples/artifact_replay_demo.py
```

## Regime presets

Runnable preset demos:

```bash
python liq-evolution/examples/baseline_regime_preset_demo.py
python liq-evolution/examples/stability_first_regime_preset_demo.py
```

Documentation:

- `liq-evolution/docs/regime_program_guide.md`
