"""The contract every declared null is run against.

Tier 2. A null that cannot say what it keeps and what it breaks cannot be
interpreted, because the same number means different things depending on which
structure survived the construction. So the declaration is checked as part of
the null, not as documentation about it, and a draw is checked to carry the
series, replicate and seed that rebuild it.

Which construction a source uses is its own business; nothing here says how the
values must be arranged. What is contracted is that a draw is derived from the
real series it was handed — same length, finite values, the handed-in series
left as it was — so that nothing is invented, imputed or extended.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import ClassVar

import pytest

from liq.evolution.ecology import NullDeclaration, SurrogateSeries, SurrogateSource
from liq.evolution.ecology.adapters import NullSurrogateSource

SOURCE_ID = "contract-series"

# A declared probe series, not market data: a plain arithmetic ramp. Only its
# length and the values handed in are read back out of a draw.
OBSERVED = tuple(float(index) for index in range(16))

DRAWS = ((0, 11), (1, 11), (0, 12))
DRAW_IDS = ("first-replicate", "second-replicate", "other-seed")


class SurrogateSourceContract:
    """What a declared null promises, whichever construction it realises.

    An adapter joins the contract by subclassing this and naming a factory that
    builds it. No suite is copied to add one::

        class TestMySurrogateSource(SurrogateSourceContract):
            adapter_factory = staticmethod(MySurrogateSource)
    """

    adapter_factory: ClassVar[Callable[[], SurrogateSource]]

    @pytest.fixture
    def adapter(self) -> SurrogateSource:
        """The adapter under test, built fresh for this test."""
        return type(self).adapter_factory()

    def test_the_adapter_satisfies_the_port(self, adapter: SurrogateSource) -> None:
        """Conformance is of the object handed over, not of a declared class."""
        assert isinstance(adapter, SurrogateSource)

    def test_the_null_says_what_it_keeps_what_it_breaks_and_what_that_tests(
        self, adapter: SurrogateSource
    ) -> None:
        """An undeclared null cannot be read, so declaring it is part of it."""
        declaration = adapter.declaration()
        assert isinstance(declaration, NullDeclaration)
        assert declaration.null_id != ""
        assert declaration.preserves != ()
        assert declaration.destroys != ()
        assert set(declaration.preserves).isdisjoint(declaration.destroys)
        assert len(declaration.hypothesis.split()) > 3

    def test_the_declaration_does_not_change_between_readings(
        self, adapter: SurrogateSource
    ) -> None:
        """A null distribution is read against one declaration, not a moving one."""
        assert adapter.declaration() == adapter.declaration()

    @pytest.mark.parametrize(("replicate", "seed"), DRAWS, ids=DRAW_IDS)
    def test_a_draw_realises_the_declared_null(
        self, adapter: SurrogateSource, replicate: int, seed: int
    ) -> None:
        """A draw is attributable to the construction it came from."""
        surrogate = adapter.draw(
            OBSERVED, source_id=SOURCE_ID, replicate=replicate, seed=seed
        )
        assert isinstance(surrogate, SurrogateSeries)
        assert surrogate.null_id == adapter.declaration().null_id

    @pytest.mark.parametrize(("replicate", "seed"), DRAWS, ids=DRAW_IDS)
    def test_a_draw_carries_everything_needed_to_rebuild_it(
        self, adapter: SurrogateSource, replicate: int, seed: int
    ) -> None:
        """A null distribution is rebuilt from its record alone."""
        surrogate = adapter.draw(
            OBSERVED, source_id=SOURCE_ID, replicate=replicate, seed=seed
        )
        assert surrogate.source_id == SOURCE_ID
        assert surrogate.replicate == replicate
        assert surrogate.seed == seed

    @pytest.mark.parametrize(("replicate", "seed"), DRAWS, ids=DRAW_IDS)
    def test_a_draw_is_positionally_aligned_with_the_series_it_came_from(
        self, adapter: SurrogateSource, replicate: int, seed: int
    ) -> None:
        """Nothing is invented, imputed or extended: the length is the series'."""
        surrogate = adapter.draw(
            OBSERVED, source_id=SOURCE_ID, replicate=replicate, seed=seed
        )
        assert isinstance(surrogate.values, tuple)
        assert len(surrogate.values) == len(OBSERVED)
        assert all(math.isfinite(value) for value in surrogate.values)

    @pytest.mark.parametrize(("replicate", "seed"), DRAWS, ids=DRAW_IDS)
    def test_draws_are_reproducible(
        self, adapter: SurrogateSource, replicate: int, seed: int
    ) -> None:
        """The same series, replicate and seed give the same surrogate."""
        first = adapter.draw(
            OBSERVED, source_id=SOURCE_ID, replicate=replicate, seed=seed
        )
        second = adapter.draw(
            OBSERVED, source_id=SOURCE_ID, replicate=replicate, seed=seed
        )
        assert first == second

    def test_a_draw_leaves_the_series_it_was_handed_alone(
        self, adapter: SurrogateSource
    ) -> None:
        """The real series is read for a draw, never rearranged in place."""
        handed_over = list(OBSERVED)
        adapter.draw(handed_over, source_id=SOURCE_ID, replicate=0, seed=11)
        assert handed_over == list(OBSERVED)

    def test_a_draw_is_attributed_to_the_series_it_was_asked_about(
        self, adapter: SurrogateSource
    ) -> None:
        """Two series under one null stay distinguishable in the record."""
        other = adapter.draw(OBSERVED, source_id="other-series", replicate=0, seed=11)
        assert other.source_id == "other-series"


class TestNullSurrogateSource(SurrogateSourceContract):
    """The stand-in, held to the same contract as any provider-backed null."""

    adapter_factory = staticmethod(NullSurrogateSource)
