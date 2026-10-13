"""The ecology descriptors' store, window, refusals, drift and conformance.

What ``tests/facts/test_descriptors.py`` binds is the per-axis computation
against an independent oracle. What is checked here is the machinery under it,
as ``plans/oracle-v4-descriptor-axes-2026-10-02.md`` rules it:

* the two bounded histories the store gains (§5.1) — each agent's realised
  exposures, written by ``hold``, and the population's reference returns,
  written once per decision point by the accounting step — their bounds, their
  order, what a birth and a death do to them, and what a restore refuses;
* the window (§2.1): a founder's first exposure ages out of it, an agent with
  too few outcomes is refused by a typed error, and histories that do not line
  up are refused rather than truncated;
* the construction refusals (§1.1, §2.1, §2.4, §4.1) and the bins left to the
  descriptor-parameter ruling (§6.3), which have no default here;
* the drift reading (§6.1);
* conformance with the GP stream's public extractor (§3.2(1)): the ecology's
  turnover and horizon are, by definition, its ``turnover`` and
  ``holding_period_proxy`` raw values over the same exposure trace. This case
  reaches ``liq.sim`` through that extractor, which a test may and the ecology
  may not.

Nothing here is market data: every stream is declared arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from liq.evolution import ecology
from liq.evolution.config import FitnessConfig
from liq.evolution.ecology import agent, descriptors
from liq.evolution.ecology.adapters import NullExecutionSimulator, NullRiskSizer
from liq.evolution.ecology.config import LearningConfig, PopulationConfig
from liq.evolution.ecology.errors import EcologyError
from liq.evolution.ecology.types import Descriptor
from liq.evolution.fitness.behavior_descriptors import extract_behavior_descriptors
from liq.evolution.fitness.evaluation_schema import (
    BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
    BEHAVIOR_DESCRIPTOR_TURNOVER,
)
from liq.gp.evolution.qd_archive import QDArchive
from tests.support import turnover_stream as streams

#: The descriptor window the walks here are read over.
WINDOW = 6

#: How many outcomes the store keeps in the walks here: room for the window
#: and more, so the window is a tail and not the whole history.
CAPACITY = 16

#: How many decision points the walks here cover.
DECISION_POINTS = 40


def _mandate(**overrides: float) -> descriptors.Mandate:
    declared = {
        "max_position": 1.0,
        "max_gross_exposure": NullRiskSizer().max_gross_exposure,
        "max_weight_delta": 2.0,
        "max_turnover": 1.0,
    }
    declared.update(overrides)
    return descriptors.Mandate(**declared)


def _declaration(**overrides: object) -> descriptors.DescriptorDeclaration:
    declared: dict[str, object] = {
        "window": WINDOW,
        "history_capacity": CAPACITY,
        "asset_classes": ("equity",),
        "universe": {streams.INSTRUMENT: "equity"},
        "mandate": _mandate(),
    }
    declared.update(overrides)
    return descriptors.DescriptorDeclaration(**declared)


def _founded(count: int = 4, capacity: int = CAPACITY) -> agent.PopulationState:
    return agent.PopulationState.founded(
        streams.founders(count),
        population=PopulationConfig(history_capacity=capacity),
        learning=LearningConfig(),
    )


def _accounts(state: agent.PopulationState):
    return ecology.population_accounts.PopulationAccounts(
        state,
        instrument=streams.INSTRUMENT,
        sizer=NullRiskSizer(),
        simulator=NullExecutionSimulator(),
        opening_equity=1.0,
    )


def _walked(
    count: int = DECISION_POINTS, capacity: int = CAPACITY
) -> tuple[agent.PopulationState, tuple]:
    state = _founded(capacity=capacity)
    accounts = _accounts(state)
    windows = streams.priced(count=count)
    for window in windows:
        accounts.visit(window)
    return state, windows


# ---------------------------------------------------------------------------
# The exposure history
# ---------------------------------------------------------------------------


def test_hold_appends_to_a_bounded_exposure_history_oldest_first() -> None:
    state = _founded(count=2, capacity=3)
    first = state.agent_ids()[0]
    assert state.exposures(first) == ()
    for exposure in (0.1, 0.2, 0.3, 0.4, 0.5):
        state.hold(first, exposure)
    assert state.exposures(first) == (0.3, 0.4, 0.5)
    assert state.realised(first) == 0.5
    assert state.exposures(state.agent_ids()[1]) == ()


def _held_by_hand(state: agent.PopulationState, exposures: tuple[float, ...]) -> None:
    """Hold each exposure for every agent, scaled by its row, outside any walk."""
    for row, agent_id in enumerate(state.agent_ids()):
        for exposure in exposures:
            state.hold(agent_id, exposure * (row + 1))


def test_a_death_leaves_every_other_agents_histories_where_they_were() -> None:
    state = _founded(count=4)
    _held_by_hand(state, (0.1, 0.2, 0.3))
    first, *rest = state.agent_ids()
    kept = {agent_id: state.exposures(agent_id) for agent_id in rest}
    assert len(set(kept.values())) == len(rest)
    state.cull(first, died_at=streams.priced(count=2)[-1].as_of)
    assert {agent_id: state.exposures(agent_id) for agent_id in rest} == kept


def test_a_birth_starts_with_no_exposure_history() -> None:
    state = _founded(count=2)
    _held_by_hand(state, (0.1, 0.2))
    newborn = replace(
        streams.founder(99),
        agent_id="newborn",
        parents=(state.agent_ids()[0],),
        born_at=streams.priced(count=2)[-1].as_of,
    )
    state.admit(newborn)
    assert state.exposures("newborn") == ()
    assert state.history("newborn") == ()


# ---------------------------------------------------------------------------
# The reference-return history
# ---------------------------------------------------------------------------


def test_the_accounting_step_writes_one_reference_return_per_decision_point() -> None:
    """None at the first decision point, then the close-to-close return at each."""
    state, windows = _walked(count=10)
    closes = [window.bars[streams.INSTRUMENT].close for window in windows]
    assert state.reference_returns() == tuple(
        later / earlier - 1.0
        for earlier, later in zip(closes, closes[1:], strict=False)
    )
    assert len(state.reference_returns()) == len(windows) - 1


def test_the_reference_history_is_bounded_by_the_history_capacity() -> None:
    state, windows = _walked(count=10, capacity=4)
    closes = [window.bars[streams.INSTRUMENT].close for window in windows]
    expected = tuple(
        later / earlier - 1.0
        for earlier, later in zip(closes, closes[1:], strict=False)
    )
    assert state.reference_returns() == expected[-4:]


def test_a_lone_agent_evaluated_on_its_own_accepts_the_reference_and_keeps_nothing() -> (
    None
):
    """``evaluate`` drives a population of one; the reference changes nothing there."""
    member = ecology.accounts._OneMember(agent=object())  # ty: ignore[invalid-argument-type]
    assert member.record_reference_return(0.01) is None


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def test_a_founders_first_exposure_ages_out_of_the_window() -> None:
    """A founder holds one exposure more than it has outcomes; the window is the last W."""
    state, _ = _walked(count=WINDOW + 1)
    for agent_id in state.agent_ids():
        assert len(state.history(agent_id)) == WINDOW
        assert len(state.exposures(agent_id)) == WINDOW + 1
        described = descriptors.describe(state, _declaration(), agent_id)
        tail = state.exposures(agent_id)[-WINDOW:]
        deltas = [abs(b - a) for a, b in zip(tail, tail[1:], strict=False)]
        assert described.raw["turnover"] == sum(deltas) / len(deltas)


def test_too_few_outcomes_raise_a_typed_refusal_naming_what_was_held() -> None:
    state, _ = _walked(count=WINDOW)
    agent_id = state.agent_ids()[0]
    with pytest.raises(descriptors.DescriptorUnavailable) as refused:
        descriptors.describe(state, _declaration(), agent_id)
    assert isinstance(refused.value, EcologyError)
    assert (refused.value.agent_id, refused.value.held, refused.value.window) == (
        agent_id,
        WINDOW - 1,
        WINDOW,
    )
    message = str(refused.value)
    assert agent_id in message and str(WINDOW - 1) in message and str(WINDOW) in message
    reading = descriptors.read(state, _declaration())
    assert reading.described == {}
    assert set(reading.undescribed) == set(state.agent_ids())
    assert reading.window == WINDOW


def test_more_outcomes_than_reference_returns_are_refused_not_truncated() -> None:
    state, _ = _walked(count=WINDOW + 2)
    put_away = state.snapshot()
    short = replace(put_away, reference_returns=put_away.reference_returns[:2])
    with pytest.raises(descriptors.DescriptorsMisaligned, match="reference"):
        descriptors.read(agent.PopulationState.restored(short), _declaration())


def test_fewer_exposures_than_outcomes_are_refused_not_truncated() -> None:
    state, _ = _walked(count=WINDOW + 2)
    put_away = state.snapshot()
    first = put_away.agents[0]
    short = replace(
        put_away,
        agents=(replace(first, exposures=first.exposures[:2]), *put_away.agents[1:]),
    )
    with pytest.raises(descriptors.DescriptorsMisaligned, match="exposure"):
        descriptors.describe(
            agent.PopulationState.restored(short), _declaration(), first.agent_id
        )


def test_a_restore_refuses_histories_longer_than_its_ceiling() -> None:
    state, _ = _walked(count=WINDOW + 2)
    put_away = state.snapshot()
    first = put_away.agents[0]
    too_many = tuple(0.0 for _ in range(CAPACITY + 1))
    with pytest.raises(ValueError, match="exposures"):
        agent.PopulationState.restored(
            replace(
                put_away,
                agents=(replace(first, exposures=too_many), *put_away.agents[1:]),
            )
        )
    with pytest.raises(ValueError, match="reference returns"):
        agent.PopulationState.restored(replace(put_away, reference_returns=too_many))


# ---------------------------------------------------------------------------
# What a described agent carries
# ---------------------------------------------------------------------------


def test_a_described_agent_converts_to_the_ports_descriptor_in_axis_order() -> None:
    state, _ = _walked()
    described = descriptors.describe(state, _declaration(), state.agent_ids()[0])
    as_port = described.descriptor()
    assert isinstance(as_port, Descriptor)
    assert tuple(as_port.values) == descriptors.AXES
    assert as_port.schema_version == descriptors.VOCABULARY == "ecology-descriptors-2"
    assert dict(as_port.values) == dict(described.values)
    assert described.values["asset_class"] == 0.5
    assert described.raw["asset_class"] == 0.0


def test_the_beta_estimator_is_named_as_provenance_records_it() -> None:
    assert descriptors.BETA_ESTIMATOR == "least_squares_through_origin"
    assert "BETA_ESTIMATOR" in descriptors.__all__


def test_the_asset_class_ordinal_follows_the_declared_order() -> None:
    declared = _declaration(
        asset_classes=("equity",), universe={streams.INSTRUMENT: "equity"}
    )
    assert declared.asset_class_count == 1
    assert declared.asset_class_ordinal == 0


def test_the_derived_bounds_come_from_the_mandate() -> None:
    declared = _declaration(
        mandate=_mandate(
            max_position=0.8,
            max_gross_exposure=0.9,
            max_weight_delta=0.7,
            max_turnover=0.6,
        )
    )
    assert declared.mandate.turnover_cap == 0.6
    assert declared.mandate.beta_max == 0.9


# ---------------------------------------------------------------------------
# Construction refusals and the bins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"history_capacity": WINDOW - 1}, "history_capacity"),
        ({"window": 1, "history_capacity": 1}, "window"),
        (
            {"universe": {streams.INSTRUMENT: "equity", "OTHER": "equity"}},
            "one traded instrument",
        ),
        ({"universe": {}}, "one traded instrument"),
        ({"universe": {streams.INSTRUMENT: "fx"}}, "declares no class"),
        (
            {"asset_classes": ("equity", "fx")},
            "more than one asset class",
        ),
        ({"asset_classes": ()}, "no asset class"),
        ({"vocabulary": descriptors.PROVISIONAL_VOCABULARY}, "provisional"),
        ({"vocabulary": "ecology-descriptors-1"}, "vocabulary"),
    ],
)
def test_a_declaration_the_record_rules_out_is_refused(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _declaration(**overrides)


@pytest.mark.parametrize(
    "bound", ["max_position", "max_gross_exposure", "max_weight_delta", "max_turnover"]
)
@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_a_mandate_bound_that_is_not_positive_and_finite_is_refused(
    bound: str, value: float
) -> None:
    with pytest.raises(ValueError, match=bound):
        _mandate(**{bound: value})


def test_the_bins_have_no_default_and_follow_the_declared_range() -> None:
    declared = _declaration()
    with pytest.raises(TypeError):
        declared.archive_bins()  # ty: ignore[missing-argument]
    assert declared.archive_bins(holding_horizon=2, turnover=16, beta=5) == (
        1,
        2,
        16,
        5,
    )
    for refused in (1, 17):
        with pytest.raises(ValueError, match=r"\[2, 16\]"):
            declared.archive_bins(holding_horizon=refused, turnover=4, beta=4)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def test_drift_is_the_within_life_change_of_each_axis_between_two_readings() -> None:
    state = _founded()
    accounts = _accounts(state)
    windows = streams.priced(count=DECISION_POINTS)
    readings = []
    for index, window in enumerate(windows):
        accounts.visit(window)
        if index in (WINDOW + 10, WINDOW + 20):
            readings.append(descriptors.read(state, _declaration()))
    earlier, later = readings

    drifted = descriptors.drift(earlier, later)

    assert tuple(drifted) == descriptors.BEHAVIOURAL_AXES
    both = sorted(set(earlier.described) & set(later.described))
    assert both
    for axis, summary in drifted.items():
        deltas = np.asarray(
            [
                abs(later.described[a].values[axis] - earlier.described[a].values[axis])
                for a in both
            ]
        )
        assert summary.count == len(both)
        assert summary.median == float(np.median(deltas))
        assert summary.q90 == float(np.quantile(deltas, 0.9))
        assert summary.maximum == float(deltas.max())


def test_drift_leaves_out_an_agent_not_described_at_both_or_excluded_on_the_axis() -> (
    None
):
    state, _ = _walked()
    reading = descriptors.read(state, _declaration())
    first, second, *_ = sorted(reading.described)
    excluded = replace(
        reading.described[first],
        excluded=MappingProxyType({"turnover": descriptors.Exclusion.NOT_FINITE}),
    )
    later = replace(
        reading,
        described=MappingProxyType(
            {
                first: excluded,
                **{
                    k: v
                    for k, v in reading.described.items()
                    if k not in (first, second)
                },
            }
        ),
    )
    drifted = descriptors.drift(reading, later)
    assert drifted["turnover"].count == len(reading.described) - 2
    assert drifted["beta"].count == len(reading.described) - 1


def test_drift_with_nobody_described_twice_reports_a_count_of_zero() -> None:
    state, _ = _walked(count=WINDOW)
    empty = descriptors.read(state, _declaration())
    drifted = descriptors.drift(empty, empty)
    for summary in drifted.values():
        assert summary.count == 0
        assert (
            summary.median is None and summary.q90 is None and summary.maximum is None
        )


def test_the_population_quantiles_are_read_over_the_described() -> None:
    state, _ = _walked()
    reading = descriptors.read(state, _declaration())
    quantiles = descriptors.marginal_quantiles(reading)
    assert tuple(quantiles) == descriptors.BEHAVIOURAL_AXES
    for axis, (low, middle, high) in quantiles.items():
        values = np.asarray([d.values[axis] for d in reading.described.values()])
        assert (low, middle, high) == tuple(
            float(value) for value in np.quantile(values, (0.1, 0.5, 0.9))
        )
    empty = descriptors.marginal_quantiles(
        descriptors.read(_walked(count=2)[0], _declaration())
    )
    assert all(value is None for value in empty.values())


# ---------------------------------------------------------------------------
# Beta: least squares through the origin
# ---------------------------------------------------------------------------

#: The configured fitness window: twenty decision points, where the computed
#: mean of twenty equal returns of 0.01 is not 0.01.
BETA_WINDOW = 20

#: A reference that drifts: every return positive and falling, ``r_k =
#: 0.010 - 0.0001 k`` for ``k`` in ``0..19``.
DRIFTING_REFERENCE = tuple(0.010 - 0.0001 * k for k in range(BETA_WINDOW))

#: An exposure carried flat for the first half of the window and fully for the
#: second.
HALF_CARRIED = (0.0,) * (BETA_WINDOW // 2) + (1.0,) * (BETA_WINDOW // 2)


def _archive() -> QDArchive:
    return QDArchive(
        n_dims=4,
        bins_per_dim=(1, 2, 2, 2),
        descriptor_bounds=((0.0, 1.0),) * 4,
        objective_directions=list(FitnessConfig().objective_directions)[:1],
    )


def _beta_window_described(
    outcomes: tuple[float, ...], reference: tuple[float, ...]
) -> descriptors.AgentDescriptor:
    """A walked agent whose last ``BETA_WINDOW`` outcomes and reference returns are these."""
    window = BETA_WINDOW
    state, _ = _walked(count=window + 4, capacity=window + 4)
    put_away = state.snapshot()
    subject = put_away.agents[0]
    assert len(subject.history) >= window
    assert len(put_away.reference_returns) >= window
    seeded = replace(
        put_away,
        agents=(
            replace(subject, history=(*subject.history[:-window], *outcomes)),
            *put_away.agents[1:],
        ),
        reference_returns=(*put_away.reference_returns[:-window], *reference),
    )
    return descriptors.describe(
        agent.PopulationState.restored(seeded),
        _declaration(window=window, history_capacity=window + 4),
        subject.agent_id,
    )


def _ordered(described: descriptors.AgentDescriptor) -> tuple[float, ...]:
    return tuple(described.values[axis] for axis in descriptors.AXES)


def test_beta_on_a_drifting_reference_is_the_carried_exposure_share() -> None:
    """Carried for half a drifting window, beta is that half's share of ``Σ r²``.

    The outcomes are the carried exposure times the reference's return, so the
    slope through the origin is ``Σ_{k≥10} r_k² / Σ r_k²``: the ``r²``-weighted
    mean of the exposure, inside [0, 1]. An intercept would remove the drift
    from the move and read the timing of the exposure instead.
    """
    outcomes = tuple(
        held * move for held, move in zip(HALF_CARRIED, DRIFTING_REFERENCE, strict=True)
    )
    expected = math.fsum(
        move * move for move in DRIFTING_REFERENCE[BETA_WINDOW // 2 :]
    ) / math.fsum(move * move for move in DRIFTING_REFERENCE)

    described = _beta_window_described(outcomes, DRIFTING_REFERENCE)

    assert expected == pytest.approx(0.4449747674, abs=1e-10)
    assert described.raw["beta"] == pytest.approx(expected, rel=1e-12)
    assert described.values["beta"] == described.raw["beta"]
    assert "beta" not in described.excluded


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ((0.01,) * BETA_WINDOW, 1.0),
        (tuple(0.001 * k for k in range(BETA_WINDOW)), 0.95),
    ],
    ids=["outcomes-equal-the-reference", "outcomes-rising"],
)
def test_beta_on_a_constant_nonzero_reference_is_the_mean_outcome_over_the_return(
    outcomes: tuple[float, ...], expected: float
) -> None:
    """A reference that moved by the same 0.01 every time is a defined move.

    Through the origin, beta is ``mean(r_a) / 0.01``: the account's carried
    fraction of that move, binnable and not excluded.
    """
    described = _beta_window_described(outcomes, (0.01,) * BETA_WINDOW)

    assert described.raw["beta"] == pytest.approx(expected)
    assert described.values["beta"] == pytest.approx(expected)
    assert not described.excluded
    assert _archive().insert("subject", (0.0,), _ordered(described))


def test_beta_is_undefined_on_an_all_zero_reference() -> None:
    """A reference window every return of which is exactly zero has no move to carry.

    Raw and normalised are both ``nan``, excluded as ``undefined``, and refused
    by the archive rather than binned.
    """
    described = _beta_window_described(
        tuple(0.001 * k for k in range(BETA_WINDOW)), (0.0,) * BETA_WINDOW
    )

    assert math.isnan(described.raw["beta"])
    assert math.isnan(described.values["beta"])
    assert described.excluded == {"beta": descriptors.Exclusion.UNDEFINED}
    with pytest.raises(ValueError, match="outside bounds"):
        _archive().insert("subject", (0.0,), _ordered(described))


@pytest.mark.parametrize("exposure", [0.0, 0.25, 1.0])
def test_beta_on_a_constant_exposure_window_is_that_exposure(exposure: float) -> None:
    """An account that carried one exposure throughout has exactly that beta.

    On the drifting reference, outcomes ``e · r`` give ``e`` whatever the drift.
    """
    outcomes = tuple(exposure * move for move in DRIFTING_REFERENCE)

    described = _beta_window_described(outcomes, DRIFTING_REFERENCE)

    assert described.raw["beta"] == pytest.approx(exposure, abs=1e-12)


# ---------------------------------------------------------------------------
# Conformance with the GP stream's public extractor
# ---------------------------------------------------------------------------


def test_turnover_and_horizon_equal_the_gp_extractors_raw_values() -> None:
    """One semantics per descriptor name, over every walked window here.

    Exact equality on traces inside the extractor's ranges (horizon at most
    512, turnover at most 5), which every window at most ``history_capacity``
    long and every long-only trace satisfies. A change to either side's
    definition turns this red.
    """
    state = _founded(count=6)
    accounts = _accounts(state)
    compared = 0
    for window in streams.priced(count=DECISION_POINTS):
        accounts.visit(window)
        for agent_id, described in descriptors.read(
            state, _declaration()
        ).described.items():
            trace = list(state.exposures(agent_id)[-WINDOW:])
            profile = extract_behavior_descriptors(
                {"position_trace": trace},
                descriptor_names=(
                    BEHAVIOR_DESCRIPTOR_TURNOVER,
                    BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
                ),
            )
            assert (
                described.raw["turnover"] == profile.raw[BEHAVIOR_DESCRIPTOR_TURNOVER]
            )
            assert (
                described.raw["holding_horizon"]
                == profile.raw[BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY]
            )
            compared += 1
    assert compared > 50


@pytest.mark.parametrize(
    "reference",
    [
        (0.0, 1e-200) * (WINDOW // 2),
        (-1e200, 1e200) * (WINDOW // 2),
        (0.01, math.nan) * (WINDOW // 2),
    ],
    ids=["denominator-underflows", "denominator-overflows", "reference-not-finite"],
)
@pytest.mark.filterwarnings("ignore:overflow encountered in square:RuntimeWarning")
def test_beta_is_undefined_where_the_denominator_is_not_a_usable_divisor(
    reference: tuple[float, ...],
) -> None:
    """Distinct reference returns whose squared spread underflows to zero or overflows.

    The reference is not constant, so only the quotient's own guard can bind:
    the IEEE quotient is undefined, and so is beta. A reference return that is
    not finite makes beta undefined before any quotient is taken.
    """
    state, _ = _walked(count=WINDOW + 2)
    put_away = state.snapshot()
    seeded = replace(
        put_away,
        reference_returns=(*put_away.reference_returns[:-WINDOW], *reference),
    )
    for described in descriptors.read(
        agent.PopulationState.restored(seeded), _declaration()
    ).described.values():
        assert math.isnan(described.raw["beta"])
        assert math.isnan(described.values["beta"])
        assert described.excluded["beta"] == descriptors.Exclusion.UNDEFINED
