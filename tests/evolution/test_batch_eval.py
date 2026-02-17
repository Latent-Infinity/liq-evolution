"""Tests for batch/mini-batch evaluation config wiring."""

from __future__ import annotations

import pytest

from liq.evolution.config import EvolutionConfig, build_gp_config
from liq.evolution.errors import ConfigurationError


class TestBatchConfigDefaults:
    """Verify default batch evaluation settings."""

    def test_batch_size_none_by_default(self) -> None:
        config = EvolutionConfig()
        assert config.batch_size is None

    def test_full_eval_interval_default_is_10(self) -> None:
        config = EvolutionConfig()
        assert config.full_eval_interval == 10


class TestBatchConfigWiring:
    """Verify batch settings wire through build_gp_config to liq-gp."""

    def test_batch_size_wires_to_gp_fitness_config(self) -> None:
        config = EvolutionConfig(batch_size=64)
        gp = build_gp_config(config)
        assert gp.fitness.batch_size == 64

    def test_batch_size_none_wires_as_none(self) -> None:
        config = EvolutionConfig(batch_size=None)
        gp = build_gp_config(config)
        assert gp.fitness.batch_size is None

    def test_full_eval_interval_wires_through(self) -> None:
        config = EvolutionConfig(full_eval_interval=5)
        gp = build_gp_config(config)
        assert gp.fitness.full_eval_interval == 5

    def test_full_eval_interval_default_wires_through(self) -> None:
        config = EvolutionConfig()
        gp = build_gp_config(config)
        assert gp.fitness.full_eval_interval == 10

    def test_batch_size_one_accepted(self) -> None:
        config = EvolutionConfig(batch_size=1)
        gp = build_gp_config(config)
        assert gp.fitness.batch_size == 1


class TestBatchConfigValidation:
    """Verify invalid batch settings are rejected."""

    def test_batch_size_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="batch_size"):
            EvolutionConfig(batch_size=0)

    def test_batch_size_negative_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="batch_size"):
            EvolutionConfig(batch_size=-1)

    def test_full_eval_interval_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="full_eval_interval"):
            EvolutionConfig(full_eval_interval=0)

    def test_full_eval_interval_negative_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="full_eval_interval"):
            EvolutionConfig(full_eval_interval=-1)
