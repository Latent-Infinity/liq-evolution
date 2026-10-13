"""Surface contract of the ecology's ports and the values they carry.

Tier 2. These checks guard the boundary itself rather than any behaviour behind
it: that the package states what it is, that every port and value type it
defines is reachable from one import, that the ports can be structurally
checked by the adapter contract suites, and that a value crossing a boundary
cannot be mutated by one reader under another.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from liq.evolution import ecology
from liq.evolution.ecology import ports, types

PORT_NAMES = (
    "AgentPopulation",
    "BarSource",
    "ExecutionSimulator",
    "FitnessDeflator",
    "RiskSizer",
    "SurrogateSource",
)

TYPE_ALIASES = (
    "AgentId",
    "CostScenarioId",
    "InstrumentId",
    "SegmentRole",
    "Standing",
    "UtcTimestamp",
)

VALUE_TYPES = tuple(
    name for name in types.__all__ if dataclasses.is_dataclass(getattr(types, name))
)

#: Values the boundary must carry, named rather than discovered, so a value the
#: ports depend on cannot drop out of the vocabulary without a check noticing.
#: A birth hands back an ``Offspring`` — the child with the variation that made
#: it — rather than a bare genome; a deflation takes ``FitnessSample`` values
#: and hands back ``DeflatedFitness`` ones.
REQUIRED_VALUE_TYPES = ("DeflatedFitness", "FitnessSample", "Genome", "Offspring")


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


def test_the_six_ports_are_the_declared_boundary() -> None:
    """Exactly the six named capabilities are exposed as ports."""
    assert set(ports.__all__) == set(PORT_NAMES)


def test_value_types_and_aliases_account_for_the_whole_vocabulary() -> None:
    """Every exported name is either a boundary value or a named identity."""
    assert set(types.__all__) == set(VALUE_TYPES) | set(TYPE_ALIASES)


@pytest.mark.parametrize("name", REQUIRED_VALUE_TYPES)
def test_the_values_the_ports_depend_on_are_in_the_vocabulary(name: str) -> None:
    """Each named value is exported, reachable from the package, and frozen data."""
    assert name in types.__all__
    assert dataclasses.is_dataclass(getattr(ecology, name))


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


def _child() -> types.Genome:
    """A two-gene child, named for the report cases below and nothing else."""
    return types.Genome(
        genes=MappingProxyType({"first_gene": 0.0, "second_gene": 1.0}),
        schema_version="surface-genes-1",
    )


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan])
def test_a_birth_cannot_report_a_scale_that_is_not_one(scale: float) -> None:
    """A scale in force is positive and finite; no scale at all is ``None``."""
    with pytest.raises(ValueError, match="positive and finite"):
        types.Offspring(
            genome=_child(),
            mutation_scale=scale,
            boundary_firings=MappingProxyType({"first_gene": 0, "second_gene": 0}),
        )


@pytest.mark.parametrize(
    "firings",
    [{"first_gene": 0}, {"first_gene": 0, "second_gene": 0, "third_gene": 0}],
)
def test_a_birth_reports_firings_for_exactly_the_genes_it_carries(
    firings: dict[str, int],
) -> None:
    """A missing count could be read as a rule that did not fire; it is refused."""
    with pytest.raises(ValueError, match="exactly the genes"):
        types.Offspring(
            genome=_child(),
            mutation_scale=None,
            boundary_firings=MappingProxyType(firings),
        )


def test_a_birth_cannot_report_a_negative_firing_count() -> None:
    """A rule cannot have acted fewer than no times."""
    with pytest.raises(ValueError, match="cannot be negative"):
        types.Offspring(
            genome=_child(),
            mutation_scale=None,
            boundary_firings=MappingProxyType({"first_gene": 0, "second_gene": -1}),
        )


def test_a_deflation_verdict_is_one_of_the_three_a_deflator_may_give() -> None:
    """A fourth verdict would be one no reader knows how to cull or keep on."""
    with pytest.raises(ValueError, match="worse, indistinguishable or better"):
        types.DeflatedFitness(
            agent_id="surface-agent",
            undeflated=0.0,
            deflated=0.0,
            compared=1,
            independent_outcomes=2,
            standing="distinguishable",  # ty: ignore[invalid-argument-type]
        )


def _verdict(
    *,
    undeflated: float | None,
    deflated: float | None,
    standing: types.Standing = "indistinguishable",
) -> types.DeflatedFitness:
    """One deflation result with only its figures and standing chosen."""
    return types.DeflatedFitness(
        agent_id="surface-agent",
        undeflated=undeflated,
        deflated=deflated,
        compared=1,
        independent_outcomes=2,
        standing=standing,
    )


@pytest.mark.parametrize("figure", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("which", ["undeflated", "deflated"])
def test_a_present_figure_must_be_finite(which: str, figure: float) -> None:
    """A figure that is present is a number; absence is ``None``, never a NaN."""
    figures: dict[str, float | None] = {"undeflated": 0.0, "deflated": 0.0}
    figures[which] = figure
    with pytest.raises(ValueError, match="finite"):
        _verdict(undeflated=figures["undeflated"], deflated=figures["deflated"])


@pytest.mark.parametrize("standing", ["worse", "better"])
def test_an_unassessed_agent_is_told_apart_from_nobody(
    standing: types.Standing,
) -> None:
    """With no deflated figure there is no verdict, so no direction either."""
    with pytest.raises(ValueError, match="indistinguishable"):
        _verdict(undeflated=0.0, deflated=None, standing=standing)


def test_nothing_is_deflated_from_a_figure_that_does_not_exist() -> None:
    """A deflated figure without an undeflated one would come from nowhere."""
    with pytest.raises(ValueError, match="undeflated"):
        _verdict(undeflated=None, deflated=0.0)


@pytest.mark.parametrize(
    ("undeflated", "deflated", "assessed"),
    [(0.5, 0.25, True), (0.5, None, False), (None, None, False)],
)
def test_a_result_is_assessed_exactly_when_it_carries_a_deflated_figure(
    undeflated: float | None, deflated: float | None, assessed: bool
) -> None:
    """``assessed`` reads the deflated figure's presence and nothing else."""
    assert _verdict(undeflated=undeflated, deflated=deflated).assessed is assessed
