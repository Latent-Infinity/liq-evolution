"""Surface contract of the ecology's ports and the values they carry.

Tier 2. These checks guard the boundary itself rather than any behaviour behind
it: that the package states what it is, that every port and value type it
defines is reachable from one import, that the ports can be structurally
checked by the adapter contract suites, and that a value crossing a boundary
cannot be mutated by one reader under another.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from liq.evolution import ecology
from liq.evolution.ecology import ports, types

PORT_NAMES = (
    "AgentPopulation",
    "BarSource",
    "ExecutionSimulator",
    "RiskSizer",
    "SurrogateSource",
)

TYPE_ALIASES = (
    "AgentId",
    "CostScenarioId",
    "InstrumentId",
    "SegmentRole",
    "UtcTimestamp",
)

VALUE_TYPES = tuple(
    name for name in types.__all__ if dataclasses.is_dataclass(getattr(types, name))
)


def test_package_states_what_it_is() -> None:
    """The package carries a real module docstring, not an empty one."""
    assert ecology.__doc__ is not None
    assert len(ecology.__doc__.strip().splitlines()) > 1


def test_every_port_and_value_type_is_reachable_from_one_import() -> None:
    """Consumers import the boundary from the package, not from its modules."""
    expected = set(ports.__all__) | set(types.__all__)
    assert set(ecology.__all__) == expected
    for name in expected:
        assert getattr(ecology, name) is not None


def test_the_five_ports_are_the_declared_boundary() -> None:
    """Exactly the five named capabilities are exposed as ports."""
    assert set(ports.__all__) == set(PORT_NAMES)


def test_value_types_and_aliases_account_for_the_whole_vocabulary() -> None:
    """Every exported name is either a boundary value or a named identity."""
    assert set(types.__all__) == set(VALUE_TYPES) | set(TYPE_ALIASES)


@pytest.mark.parametrize("name", PORT_NAMES)
def test_ports_are_structurally_checkable_protocols(name: str) -> None:
    """A port is satisfied by shape, and conformance is checkable at runtime."""
    port = getattr(ports, name)
    with pytest.raises(TypeError):
        port()
    assert not isinstance(object(), port)


@pytest.mark.parametrize("name", VALUE_TYPES)
def test_boundary_values_refuse_mutation(name: str) -> None:
    """One reader cannot change what another sees within a decision point.

    The probe is built without running ``__init__``, so that a value which
    checks its own invariants is held to the same property as one which does
    not. Refusing mutation belongs to the class, not to any particular
    well-formed instance of it, and probing it through a constructor would
    make the check depend on knowing what each value considers well formed.
    """
    value_type = getattr(types, name)
    fields = dataclasses.fields(value_type)
    instance = object.__new__(value_type)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, fields[0].name, None)


def test_an_outcome_that_permits_nothing_must_say_why() -> None:
    """Withholding action for no recorded reason cannot be expressed at all."""
    with pytest.raises(ValueError, match="must name at least one rejection"):
        types.SizingOutcome(target=None, rejections=())


def test_an_outcome_that_traded_nothing_must_say_why() -> None:
    """The same discipline at the other boundary: a non-fill is never silent."""
    with pytest.raises(ValueError, match="must name a stable reason"):
        types.NotFilled(
            agent_id="surface-agent",
            instrument="SURFACE-A",
            as_of=datetime(2000, 1, 1, tzinfo=UTC),
            requested_exposure=0.5,
            held_exposure=0.0,
            reason="",
            cost_scenario_id="surface-scenario",
        )
