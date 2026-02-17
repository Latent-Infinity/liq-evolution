"""Exception hierarchy for liq-evolution.

All exceptions inherit from :class:`LiqEvolutionError`.  Consumers can catch
``LiqEvolutionError`` for a blanket handler or individual subclasses for
fine-grained control.

This hierarchy is independent from liq-gp's :class:`~liq.gp.errors.GPError`
so domain-layer errors can be caught separately from engine errors.
"""

from __future__ import annotations


class LiqEvolutionError(Exception):
    """Base exception for all liq-evolution errors.

    All liq-evolution exceptions subclass this, so ``except LiqEvolutionError``
        acts as a catch-all for domain-layer errors.
    """


class EvolutionError(LiqEvolutionError):
    """General domain error used by phase-0 documentation contract."""


class PrimitiveError(EvolutionError):
    """Raised when primitive registration or lookup fails."""


class EvaluationError(EvolutionError):
    """Raised when GP program evaluation or fitness pipeline execution fails.

    Trigger conditions:

    - A GP program produces invalid array shapes during evaluation.
    - Runtime errors in evolved expressions (e.g. division by zero propagation).
    - NaN propagation renders the evaluation context unusable.
    """


class SerializationError(EvolutionError):
    """Raised when program or genome serialization/deserialization fails.

    Trigger conditions:

    - Unknown node types encountered during deserialization.
    - Genome schema version mismatch or unsupported version.
    - Corrupt payload data (missing required fields, invalid structure).
    - Missing registry primitives needed to reconstruct a program.
    """


class FitnessError(EvolutionError):
    """Raised when fitness metric computation produces invalid results.

    Trigger conditions:

    - All-NaN fitness values from an objective function.
    - Label and prediction array shape mismatch.
    - Objective function domain errors (e.g. log of non-positive values).
    """


class PrimitiveSetupError(PrimitiveError):
    """Raised on indicator backend or primitive registration failures.

    Trigger conditions:

    - The indicator backend is unavailable or misconfigured.
    - A trading primitive fails to register in the GP registry.
    - A required indicator category has no available implementations.
    """


class FitnessEvaluationError(FitnessError):
    """Raised when domain fitness evaluation fails.

    Trigger conditions:

    - Label-based fitness computation encounters invalid data shapes.
    - Backtest fitness evaluation raises an unrecoverable error.
    - An objective function produces only NaN or infinite values.
    """


class AdapterError(EvolutionError):
    """Raised on liq-runner or liq-signals adapter failures.

    Trigger conditions:

    - The liq-runner strategy adapter cannot translate a GP program.
    - The liq-signals provider fails to generate signals.
    - Store/cache serialization or deserialization fails.
    """


class ConfigurationError(EvolutionError):
    """Raised when liq-evolution configuration is invalid.

    Trigger conditions:

    - ``population_size < 10``, ``max_depth < 2``, ``generations < 1``
    - ``label_top_k`` not in ``(0, 1]``
    - ``backtest_top_n < 1``
    - ``max_workers < 1`` or ``memory_limit_mb < 128``

    Raised at construction time (fail-fast) so invalid configs never
    reach the evolution loop.
    """


class ParallelExecutionError(EvolutionError):
    """Raised on parallel backend failures.

    Trigger conditions:

    - The Ray cluster cannot be initialised or has disconnected.
    - A worker process raises an unrecoverable error.
    - Memory limits are exceeded during parallel evaluation.
    """
