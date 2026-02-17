"""Tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from liq.evolution.errors import (
    AdapterError,
    ConfigurationError,
    FitnessEvaluationError,
    LiqEvolutionError,
    ParallelExecutionError,
    PrimitiveSetupError,
)

ALL_SUBCLASSES = [
    PrimitiveSetupError,
    FitnessEvaluationError,
    AdapterError,
    ConfigurationError,
    ParallelExecutionError,
]


class TestExceptionHierarchy:
    """Verify the exception hierarchy structure."""

    def test_base_is_exception(self) -> None:
        assert issubclass(LiqEvolutionError, Exception)

    def test_base_not_builtin_type_error(self) -> None:
        assert not issubclass(LiqEvolutionError, TypeError)

    @pytest.mark.parametrize("exc_cls", ALL_SUBCLASSES)
    def test_subclass_of_base(self, exc_cls: type[LiqEvolutionError]) -> None:
        assert issubclass(exc_cls, LiqEvolutionError)

    @pytest.mark.parametrize("exc_cls", ALL_SUBCLASSES)
    def test_instantiable_without_args(self, exc_cls: type[LiqEvolutionError]) -> None:
        exc = exc_cls()
        assert isinstance(exc, exc_cls)

    @pytest.mark.parametrize("exc_cls", ALL_SUBCLASSES)
    def test_instantiable_with_message(self, exc_cls: type[LiqEvolutionError]) -> None:
        msg = "test error message"
        exc = exc_cls(msg)
        assert str(exc) == msg

    @pytest.mark.parametrize("exc_cls", ALL_SUBCLASSES)
    def test_catchable_as_base(self, exc_cls: type[LiqEvolutionError]) -> None:
        with pytest.raises(LiqEvolutionError):
            raise exc_cls("caught by base")

    def test_independent_from_gp_error(self) -> None:
        """liq-evolution errors must NOT inherit from liq-gp's GPError."""
        from liq.gp.errors import GPError

        assert not issubclass(LiqEvolutionError, GPError)

    def test_base_instantiable(self) -> None:
        exc = LiqEvolutionError("base error")
        assert str(exc) == "base error"
