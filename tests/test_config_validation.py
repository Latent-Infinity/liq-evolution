"""Tests for Phase 4 configuration validation hardening."""

from __future__ import annotations

import pytest

from liq.evolution.config import EvolutionConfig, ParallelConfig
from liq.evolution.errors import ConfigurationError


class TestParallelConfigNewFields:
    """Verify new Phase 4 ParallelConfig fields and defaults."""

    def test_max_in_flight_default(self) -> None:
        config = ParallelConfig()
        assert config.max_in_flight == 4

    def test_max_tasks_per_worker_default(self) -> None:
        config = ParallelConfig()
        assert config.max_tasks_per_worker == 100

    def test_memory_warn_threshold_default(self) -> None:
        config = ParallelConfig()
        assert config.memory_warn_threshold_mb == 1536

    def test_auto_fallback_default_true(self) -> None:
        config = ParallelConfig()
        assert config.auto_fallback is True

    def test_custom_values_accepted(self) -> None:
        config = ParallelConfig(
            max_in_flight=8,
            max_tasks_per_worker=50,
            memory_warn_threshold_mb=1024,
            auto_fallback=False,
        )
        assert config.max_in_flight == 8
        assert config.max_tasks_per_worker == 50
        assert config.memory_warn_threshold_mb == 1024
        assert config.auto_fallback is False


class TestParallelConfigValidation:
    """Verify invalid ParallelConfig values are rejected."""

    def test_max_in_flight_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="max_in_flight"):
            ParallelConfig(max_in_flight=0)

    def test_max_in_flight_negative_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="max_in_flight"):
            ParallelConfig(max_in_flight=-1)

    def test_max_tasks_per_worker_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="max_tasks_per_worker"):
            ParallelConfig(max_tasks_per_worker=0)

    def test_max_tasks_per_worker_negative_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="max_tasks_per_worker"):
            ParallelConfig(max_tasks_per_worker=-1)

    def test_memory_warn_threshold_negative_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="memory_warn_threshold_mb"):
            ParallelConfig(memory_warn_threshold_mb=-1)

    def test_memory_warn_threshold_zero_accepted(self) -> None:
        config = ParallelConfig(memory_warn_threshold_mb=0)
        assert config.memory_warn_threshold_mb == 0

    def test_memory_warn_exceeds_limit_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="memory_warn_threshold_mb"):
            ParallelConfig(memory_limit_mb=1024, memory_warn_threshold_mb=1024)

    def test_memory_warn_below_limit_accepted(self) -> None:
        config = ParallelConfig(memory_limit_mb=2048, memory_warn_threshold_mb=1536)
        assert config.memory_warn_threshold_mb == 1536


class TestEvolutionConfigBatchValidation:
    """Cross-check batch validation via EvolutionConfig."""

    def test_negative_population_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="population_size"):
            EvolutionConfig(population_size=-1)

    def test_zero_generations_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="generations"):
            EvolutionConfig(generations=0)

    def test_valid_config_with_all_new_fields(self) -> None:
        config = EvolutionConfig(
            batch_size=32,
            full_eval_interval=5,
            parallel=ParallelConfig(
                backend="ray",
                max_workers=4,
                max_in_flight=8,
                max_tasks_per_worker=50,
                memory_limit_mb=4096,
                memory_warn_threshold_mb=3072,
                auto_fallback=True,
            ),
        )
        assert config.batch_size == 32
        assert config.full_eval_interval == 5
        assert config.parallel.max_in_flight == 8
