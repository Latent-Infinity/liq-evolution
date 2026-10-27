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

import dataclasses
import math
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest

from liq.evolution.ecology import (
    NullDeclaration,
    SurrogateSeries,
    SurrogateSource,
    types,
)
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

    @pytest.fixture
    def provenance(self, adapter: SurrogateSource) -> Any:
        """The provenance of one draw, read by name so the base still collects."""
        surrogate = adapter.draw(OBSERVED, source_id=SOURCE_ID, replicate=0, seed=11)
        return getattr(surrogate, "provenance")  # noqa: B009

    def test_a_draw_cannot_exist_without_the_record_of_where_it_came_from(
        self, adapter: SurrogateSource
    ) -> None:
        """A series with no provenance is unconstructible, not merely absent."""
        surrogate = adapter.draw(OBSERVED, source_id=SOURCE_ID, replicate=0, seed=11)
        names = {field.name for field in dataclasses.fields(surrogate)}
        assert "provenance" in names
        with pytest.raises(TypeError):
            SurrogateSeries(  # type: ignore[call-arg]
                null_id="n", source_id="s", replicate=0, seed=0, values=(1.0,)
            )

    def test_the_provenance_names_the_series_the_draw_came_from(
        self, adapter: SurrogateSource, provenance: Any
    ) -> None:
        """The source identity on the record is the one the draw was attributed to."""
        assert provenance.source_data_id == SOURCE_ID
        assert provenance.symbols != ()
        assert list(provenance.symbols) == sorted(set(provenance.symbols))

    def test_the_provenance_names_the_period_and_the_data_hash(
        self, provenance: Any
    ) -> None:
        """Period and hash identify the data in UTC; the hash is lowercase sha256."""
        assert provenance.period_first_bar <= provenance.period_last_bar
        assert provenance.period_first_bar.utcoffset() == UTC.utcoffset(None)
        assert re.fullmatch(r"[0-9a-f]{64}", provenance.source_data_hash)
        assert provenance.restart_indices[0] == 0

    def test_the_provenance_names_the_method_and_the_generator(
        self, provenance: Any
    ) -> None:
        """Method, generator version and the hash of each function that shaped it."""
        assert provenance.method in {"stationary_bootstrap", "iaaft", "permutation"}
        assert provenance.generator_version != ""
        assert provenance.generator_functions != ()
        names = [function.name for function in provenance.generator_functions]
        assert names == sorted(names)
        for function in provenance.generator_functions:
            assert re.fullmatch(r"[0-9a-f]{64}", function.sha256)

    @pytest.mark.parametrize(("replicate", "seed"), DRAWS, ids=DRAW_IDS)
    def test_the_provenance_carries_the_seed_the_draw_was_handed(
        self, adapter: SurrogateSource, replicate: int, seed: int
    ) -> None:
        """The seed given to the draw and the integer it derived are both recorded."""
        surrogate = adapter.draw(
            OBSERVED, source_id=SOURCE_ID, replicate=replicate, seed=seed
        )
        record = getattr(surrogate, "provenance")  # noqa: B009
        assert surrogate.seed == seed
        assert isinstance(record.generator_seed, int)
        assert isinstance(record.root_seed, int)
        assert re.fullmatch(r"[0-9a-f]{16}", record.cell_key)
        assert record.null_replicates >= 1
        assert record.instrument != ""

    def test_the_provenance_carries_the_method_parameters_its_method_needs(
        self, provenance: Any
    ) -> None:
        """Each method records its own parameters and leaves the others absent."""
        if provenance.method == "stationary_bootstrap":
            assert provenance.block_length is not None
            assert provenance.block_length_derivation
            assert provenance.row_plan is not None
            assert provenance.max_iters is None
            assert provenance.converged is None
        elif provenance.method == "iaaft":
            assert provenance.max_iters is not None
            assert isinstance(provenance.converged, bool)
            assert provenance.block_length is None
            assert provenance.row_plan is None
        else:
            for name in (
                "block_length",
                "block_length_derivation",
                "row_plan",
                "max_iters",
                "converged",
                "iterations",
                "spectral_distance",
            ):
                assert getattr(provenance, name) is None

    def test_the_provenance_carries_the_declaration_it_was_drawn_under(
        self, adapter: SurrogateSource, provenance: Any
    ) -> None:
        """The text printed beside the distribution is the object compared."""
        assert provenance.declaration == adapter.declaration()
        assert isinstance(getattr(adapter.declaration(), "limitations"), tuple)  # noqa: B009

    def test_unknown_provenance_method_is_refused(self, provenance: Any) -> None:
        with pytest.raises(ValueError, match="method"):
            dataclasses.replace(provenance, method="unknown")

    def test_the_provenance_carries_the_authority_and_the_standing_of_the_draw(
        self, provenance: Any
    ) -> None:
        """A generated draw says by whose authority it exists and that it is no signal."""
        authorized_by = getattr(types, "AUTHORIZED_BY")  # noqa: B009
        assert provenance.authorized_by == authorized_by
        assert "V0B.5" in authorized_by
        assert "agent-ecology-governance-authorization.md" in authorized_by
        assert provenance.generated_label == getattr(types, "GENERATED_LABEL")  # noqa: B009
        assert provenance.signal_bearing is False
        assert provenance.citable is False

    def test_the_provenance_declares_the_booked_read_it_derives_from(
        self, provenance: Any
    ) -> None:
        """The identity of the read travels with every draw."""
        assert provenance.source_dataset_id != ""
        assert provenance.source_read_purpose != ""
        assert provenance.source_read_arm_id != ""


def _record(**changes: Any) -> Any:
    """A well-formed provenance, from a draw, with ``changes`` applied."""
    draw = NullSurrogateSource().draw(
        OBSERVED, source_id=SOURCE_ID, replicate=0, seed=1
    )
    base = getattr(draw, "provenance")  # noqa: B009
    return dataclasses.replace(base, **changes)


def _function(**changes: Any) -> Any:
    base = _record().generator_functions[0]
    return dataclasses.replace(base, **changes)


class TestProvenanceInvariants:
    """What the record refuses to say, by field."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("source_data_hash", "A" * 64),
            ("source_data_hash", "a" * 63),
            ("cell_key", "a" * 15),
            ("cell_key", "G" * 16),
            ("generator_functions", ()),
            ("symbols", ()),
            ("symbols", ("B", "A")),
            ("symbols", ("A", "A")),
            ("restart_indices", ()),
            ("restart_indices", (1, 2)),
            ("restart_indices", (0, 0)),
            ("restart_indices", (0, -1)),
            ("null_replicates", 0),
            ("instrument", ""),
            ("generator_version", ""),
            ("period_first_bar", datetime(2100, 1, 1, tzinfo=UTC)),
            ("authorized_by", "somebody"),
            ("generated_label", "Generated"),
            ("signal_bearing", True),
            ("citable", True),
        ],
    )
    def test_an_incoherent_record_is_refused_naming_the_field(
        self, field: str, value: Any
    ) -> None:
        """Each refusal names the field it is about."""
        with pytest.raises(ValueError, match=field):
            _record(**{field: value})

    def test_a_function_hash_must_be_lowercase_sha256(self) -> None:
        """The hash of a generator function is checked as the data hash is."""
        with pytest.raises(ValueError, match="sha256"):
            _record(generator_functions=(_function(sha256="xyz"),))

    def test_a_naive_period_is_refused(self) -> None:
        """Timestamps are normalised to UTC and never guessed."""
        with pytest.raises(ValueError, match="timezone-aware"):
            _record(period_last_bar=datetime(2100, 1, 1))

    def test_a_period_in_another_zone_is_held_in_utc(self) -> None:
        """Normalisation happens at construction."""
        from datetime import timedelta, timezone

        zone = timezone(timedelta(hours=5))
        record = _record(period_last_bar=datetime(2100, 1, 1, tzinfo=zone))
        assert record.period_last_bar.utcoffset() == UTC.utcoffset(None)

    def test_a_permutation_carries_no_method_specific_field(self) -> None:
        """A permutation with a block length is a different construction."""
        with pytest.raises(ValueError, match="block_length"):
            _record(block_length=3.0)

    def test_a_series_whose_record_names_another_source_is_refused(self) -> None:
        """The record and the series cannot disagree on what was drawn from."""
        with pytest.raises(ValueError, match="source_id"):
            SurrogateSeries(
                null_id="null-order-permutation",
                source_id="elsewhere",
                replicate=0,
                seed=1,
                values=OBSERVED,
                provenance=_record(),
            )

    def test_a_series_whose_record_names_another_null_is_refused(self) -> None:
        """The record's declaration is the null the series claims."""
        with pytest.raises(ValueError, match="null_id"):
            SurrogateSeries(
                null_id="another-null",
                source_id=SOURCE_ID,
                replicate=0,
                seed=1,
                values=OBSERVED,
                provenance=_record(),
            )

    def test_a_row_plan_must_be_as_long_as_the_values(self) -> None:
        """A plan into the pool addresses exactly the rows it produced."""
        record = _record(
            method="stationary_bootstrap",
            block_length=2.0,
            block_length_derivation="declared",
            row_plan=(0, 1),
        )
        with pytest.raises(ValueError, match="row_plan"):
            SurrogateSeries(
                null_id="null-order-permutation",
                source_id=SOURCE_ID,
                replicate=0,
                seed=1,
                values=OBSERVED,
                provenance=record,
            )

    @pytest.mark.parametrize(
        "changes",
        [
            {"block_length": math.nan},
            {"block_length": 0.5},
            {"block_length_derivation": ""},
            {"row_plan": None},
            {"max_iters": 3},
            {"converged": True},
            {"iterations": 1},
            {"spectral_distance": 0.0},
        ],
    )
    def test_a_bootstrap_record_is_refused_when_incoherent(
        self, changes: dict[str, Any]
    ) -> None:
        """The stationary bootstrap records its block length and plan, nothing of IAAFT."""
        fields: dict[str, Any] = {
            "method": "stationary_bootstrap",
            "block_length": 2.0,
            "block_length_derivation": "declared",
            "row_plan": (0, 1),
        }
        fields.update(changes)
        with pytest.raises(ValueError):
            _record(**fields)

    def test_a_coherent_bootstrap_record_is_accepted(self) -> None:
        """The refusals above are not blanket."""
        record = _record(
            method="stationary_bootstrap",
            block_length=2.0,
            block_length_derivation="declared",
            row_plan=(0, 1),
        )
        assert record.row_plan == (0, 1)

    @pytest.mark.parametrize(
        "changes",
        [
            {"max_iters": 0},
            {"converged": None},
            {"iterations": 0},
            {"iterations": 9},
            {"spectral_distance": -1.0},
            {"spectral_distance": math.inf},
            {"block_length": 2.0},
            {"row_plan": (0,)},
            {"block_length_derivation": "declared"},
        ],
    )
    def test_an_iaaft_record_is_refused_when_incoherent(
        self, changes: dict[str, Any]
    ) -> None:
        """IAAFT records its iteration facts, nothing of the bootstrap."""
        fields: dict[str, Any] = {
            "method": "iaaft",
            "max_iters": 5,
            "converged": True,
            "iterations": 3,
            "spectral_distance": 0.1,
        }
        fields.update(changes)
        with pytest.raises(ValueError):
            _record(**fields)

    def test_a_coherent_iaaft_record_is_accepted(self) -> None:
        """The refusals above are not blanket."""
        record = _record(
            method="iaaft",
            max_iters=5,
            converged=False,
            iterations=5,
            spectral_distance=0.0,
        )
        assert record.converged is False


class TestNullSurrogateSource(SurrogateSourceContract):
    """The stand-in, held to the same contract as any provider-backed null."""

    adapter_factory = staticmethod(NullSurrogateSource)
