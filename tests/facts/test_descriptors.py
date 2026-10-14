"""Evidence that each ecology descriptor axis is computed as declared.

``ECOLOGY.NICHE.DESCRIPTORS_AS_DECLARED.v1`` (EV-E80). The declaration is
``plans/oracle-v4-descriptor-axes-2026-10-02.md`` §1–§2, with beta's estimator
as ``plans/oracle-v4-beta-estimator-2026-10-03.md`` §2 supersedes it, under
vocabulary ``ecology-descriptors-2``, and the fact's wording is
``plans/oracle-v4-fact-coverage-2026-10-02.md`` §2.

**The oracle is independent of the implementation.** ``tests/support/descriptor_oracle.py``
restates each axis in plain ``math`` and reads none of the store's histories:
an agent's exposures and outcomes are re-derived from the states its
evaluation account published, and the reference from the stream's own bars.
So a history written at the wrong decision point, or two windows paired out
of step, disagrees with the oracle rather than with a shifted copy of itself.

**No later read, checked absolutely.** Each reading is compared with the
oracle over histories truncated at the reading's instant, and a reading taken
mid-walk is compared again once the walk has gone on, and with the same
reading in a replay whose later bars were re-valued. A perturb-and-compare
check on its own is blind to a reader uniformly one decision point ahead;
the truncated oracle is not.

**Histories are walked, never written.** Every exposure, outcome and
reference return comes from real agents walking the declared streams (S1, the
level-shift stream with priced bars; S4, the declared ramp). The out-of-range
cases start from such a walk and seed one defect each into what the store
holds — a mandate that did not bind, a reference that never moved, a value
that is not finite — and say which defect they seed.

Nothing here is market data: every stream is declared arithmetic.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from types import ModuleType

import pytest

from liq.evolution import ecology
from liq.evolution.config import FitnessConfig
from liq.evolution.ecology import agent
from liq.evolution.ecology.adapters import NullRiskSizer
from liq.evolution.ecology.errors import EcologyError
from liq.evolution.ecology.types import BarWindow
from liq.gp.evolution.qd_archive import QDArchive
from tests.support import descriptor_oracle as oracle
from tests.support import turnover_stream as streams

#: How many decision points the walks here cover: many triggers past the
#: warm-up, so agents are born, described, and die.
DECISION_POINTS = 120

#: The descriptor window: the run's fitness window, as the record ties it.
WINDOW = 8

#: The mandate the declaration derives ``c_t`` and ``beta_max`` from. The
#: gross bound is the null sizer's own, so the bound the accounts were walked
#: under is the bound the axes are normalised by; the other three are the
#: fixture-phase values the consumer declares.
MAX_POSITION = 1.0
MAX_GROSS_EXPOSURE = NullRiskSizer().max_gross_exposure
MAX_WEIGHT_DELTA = 2.0
MAX_TURNOVER = 1.0

#: The one declared asset class of the one traded instrument (A6).
ASSET_CLASS = "equity"

#: Tolerance between the oracle's ``math.fsum`` and the implementation's
#: arithmetic: rounding, never a difference of definition.
CLOSE = {"rel": 1e-9, "abs": 1e-12}


@pytest.fixture
def descriptors() -> ModuleType:
    """The ecology descriptor module, imported per case so each one reds on its own."""
    return importlib.import_module("liq.evolution.ecology.descriptors")


def _declaration(descriptors: ModuleType, **overrides: object):
    declared: dict[str, object] = {
        "window": WINDOW,
        "history_capacity": streams.HISTORY_CAPACITY,
        "asset_classes": (ASSET_CLASS,),
        "universe": {streams.INSTRUMENT: ASSET_CLASS},
        "mandate": descriptors.Mandate(
            max_position=MAX_POSITION,
            max_gross_exposure=MAX_GROSS_EXPOSURE,
            max_weight_delta=MAX_WEIGHT_DELTA,
            max_turnover=MAX_TURNOVER,
        ),
    }
    declared.update(overrides)
    return descriptors.DescriptorDeclaration(**declared)


def _turnover():
    return ecology.steady_state.TurnoverConfig(
        band_lower=14,
        band_upper=20,
        replacement_count=2,
        trigger_period=3,
        minimum_age=6,
        replacement_window=6,
        maximum_replaced_fraction=0.3,
        fitness_window=WINDOW,
        parents_per_birth=2,
        inheritance="baldwinian",
        admission_redraws=8,
        seed=11,
    )


@dataclass
class Walked:
    """One steady-state walk and every descriptor reading taken during it."""

    windows: tuple[BarWindow, ...]
    run: streams.Steady
    readings: dict[int, object]


def _s1() -> tuple[tuple[BarWindow, ...], tuple[agent.AgentBirth, ...]]:
    return streams.priced(count=DECISION_POINTS), streams.founders(14)


def _s4() -> tuple[tuple[BarWindow, ...], tuple[agent.AgentBirth, ...]]:
    return streams.ramp(count=DECISION_POINTS), streams.ramp_founders(14)


STREAMS: dict[str, Callable[[], tuple]] = {"S1": _s1, "S4": _s4}


def _wired(founding: Sequence[agent.AgentBirth]) -> streams.Wired:
    return streams.wire(
        founding=founding,
        turnover=_turnover(),
        deflator=streams.ScriptedDeflator(
            streams.per_sample(streams.negative_is_worse)
        ),
    )


def _at_each_turnover(
    wired: streams.Wired,
    windows: Sequence[BarWindow],
    read: Callable[[int, agent.PopulationState], None],
) -> None:
    """Call ``read`` at each decision point's turnover, before its deaths and births.

    That is where the loop reads descriptors at a trigger: after the accounting
    step has written the decision point's histories, before anyone dies or is
    born there.
    """
    index_of = {window.as_of: index for index, window in enumerate(windows)}
    turn_over = wired.loop._turn_over  # ty: ignore[unresolved-attribute]

    def reading_then_turning_over(window: BarWindow) -> None:
        read(index_of[window.as_of], wired.state)
        turn_over(window)

    wired.loop._turn_over = reading_then_turning_over  # ty: ignore[unresolved-attribute]


def _walk(
    descriptors: ModuleType,
    windows: Sequence[BarWindow],
    founding: Sequence[agent.AgentBirth],
) -> Walked:
    """Walk ``windows`` with turnover, reading every living agent at each turnover."""
    readings: dict[int, object] = {}
    declaration = _declaration(descriptors)
    wired = _wired(founding)
    _at_each_turnover(
        wired,
        windows,
        lambda index, state: readings.__setitem__(
            index, descriptors.read(state, declaration)
        ),
    )
    watched = streams.Watched(windows, wired.state)
    run = streams.Steady(
        state=wired.state,
        port=wired.port,
        watched=watched,
        loop=wired.loop,
        run=wired.loop.run(watched),  # ty: ignore[unresolved-attribute]
    )
    return Walked(windows=tuple(windows), run=run, readings=readings)


def _predicted(walked: Walked, index: int, agent_id: str) -> oracle.Predicted | None:
    """The oracle's four axes for ``agent_id``, over what existed at decision point ``index``."""
    windows = walked.windows
    window = oracle.window_at(
        walked.run.run.evaluation.accounts[agent_id].states,  # ty: ignore[unresolved-attribute]
        {each.as_of: each.bars[streams.INSTRUMENT].close for each in windows},
        [each.as_of for each in windows],
        at=windows[index].as_of,
        window=WINDOW,
        instrument=streams.INSTRUMENT,
    )
    if window is None:
        return None
    return _predict(window)


def _predict(window: oracle.Window) -> oracle.Predicted:
    return oracle.predict(
        window,
        length=WINDOW,
        turnover_cap=min(
            MAX_POSITION, MAX_GROSS_EXPOSURE, MAX_WEIGHT_DELTA, MAX_TURNOVER
        ),
        beta_max=MAX_GROSS_EXPOSURE,
        class_ordinal=0,
        classes=1,
    )


def _agrees(described, predicted: oracle.Predicted) -> None:
    for axis in ("asset_class", "holding_horizon", "turnover", "beta"):
        assert described.values[axis] == pytest.approx(
            predicted.values[axis], nan_ok=True, **CLOSE
        ), (described.agent_id, axis, "normalised")
        assert described.raw[axis] == pytest.approx(
            predicted.raw[axis], nan_ok=True, **CLOSE
        ), (described.agent_id, axis, "raw")


def _held(walked: Walked, index: int, agent_id: str) -> int:
    states = walked.run.run.evaluation.accounts[agent_id].states  # ty: ignore[unresolved-attribute]
    return sum(1 for state in states if state.as_of <= walked.windows[index].as_of) - 1


def _archive() -> QDArchive:
    """The platform archive over the four axes on the unit interval.

    The bins are this check's, not the ecology's: the shipped counts are the
    descriptor-parameter ruling's
    (``plans/agent-ecology-adr-descriptor-parameters.md``), and only whether a
    value can be binned at all matters here.
    """
    return QDArchive(
        n_dims=4,
        bins_per_dim=(1, 2, 2, 2),
        descriptor_bounds=((0.0, 1.0),) * 4,
        objective_directions=list(FitnessConfig().objective_directions)[:1],
    )


def _ordered(descriptors: ModuleType, values) -> tuple[float, ...]:
    return tuple(values[axis] for axis in descriptors.AXES)


@pytest.mark.parametrize("stream", sorted(STREAMS))
def test_each_axis_equals_its_declared_computation(
    descriptors: ModuleType, stream: str
) -> None:
    """At every decision point, every described agent's four axes equal the oracle's.

    And every agent the oracle cannot describe — fewer than ``W`` outcomes
    published by then — is undescribed, holding exactly what it had.
    """
    windows, founding = STREAMS[stream]()
    walked = _walk(descriptors, windows, founding)
    assert len(walked.readings) == DECISION_POINTS

    born_later_and_described = set()
    horizons, turnovers, betas = set(), set(), set()
    founders = {birth.agent_id for birth in founding}
    for index, reading in walked.readings.items():
        living = set(walked.run.watched.living[index])
        assert set(reading.described) | set(reading.undescribed) == living
        assert not set(reading.described) & set(reading.undescribed)
        for agent_id in living:
            predicted = _predicted(walked, index, agent_id)
            if predicted is None:
                assert reading.undescribed[agent_id] == _held(walked, index, agent_id)
                continue
            described = reading.described[agent_id]
            assert described.vocabulary == descriptors.VOCABULARY
            _agrees(described, predicted)
            if agent_id not in founders:
                born_later_and_described.add(agent_id)
            horizons.add(described.values["holding_horizon"])
            turnovers.add(described.values["turnover"])
            betas.add(described.raw["beta"])

    assert born_later_and_described, (
        "no agent born during the walk was ever described, so the alignment of a "
        "newborn's histories was never put to the test"
    )
    assert len(horizons) > 1 and len(turnovers) > 1 and len(betas) > 1, (
        "an axis was constant over the walk, so agreement with the oracle says "
        f"little on it: {len(horizons)} horizons, {len(turnovers)} turnovers, "
        f"{len(betas)} betas"
    )


def _finished(descriptors: ModuleType) -> tuple[Walked, agent.PopulationSnapshot]:
    windows, founding = _s1()
    walked = _walk(descriptors, windows, founding)
    return walked, walked.run.state.snapshot()


def _described_at_the_end(descriptors: ModuleType, walked: Walked):
    reading = walked.readings[DECISION_POINTS - 1]
    return [
        described
        for described in reading.described.values()
        if not described.excluded and math.isfinite(described.raw["beta"])
    ]


def _with_agent(
    put_away: agent.PopulationSnapshot, agent_id: str, **changes: object
) -> agent.PopulationSnapshot:
    return replace(
        put_away,
        agents=tuple(
            replace(held, **changes) if held.agent_id == agent_id else held
            for held in put_away.agents
        ),
    )


def _tail(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(values[-WINDOW:])


@pytest.mark.parametrize("target", [-0.05, 2.5])
def test_a_finite_beta_outside_its_bound_is_clipped_with_the_raw_estimate_kept(
    descriptors: ModuleType, target: float
) -> None:
    """A finite beta below 0 or above ``beta_max`` is clipped; the raw estimate stays beside it.

    Seeded from a walked agent: its own outcomes, with ``(target - its own
    beta)`` times the reference added, so its residual is its own and its
    estimate lands at ``target`` — slightly negative for a long-only account
    (§D's case), or past the bound.
    """
    walked, put_away = _finished(descriptors)
    subject = _described_at_the_end(descriptors, walked)[0]
    held = next(each for each in put_away.agents if each.agent_id == subject.agent_id)
    reference = _tail(put_away.reference_returns)
    outcomes = list(held.history)
    shift = target - subject.raw["beta"]
    for offset, value in enumerate(reference):
        outcomes[len(outcomes) - WINDOW + offset] += shift * value
    seeded = agent.PopulationState.restored(
        _with_agent(put_away, subject.agent_id, history=tuple(outcomes))
    )

    described = descriptors.describe(
        seeded, _declaration(descriptors), subject.agent_id
    )

    expected_raw = oracle.beta_raw(_tail(outcomes), reference)
    assert expected_raw == pytest.approx(target, abs=1e-6)
    assert described.raw["beta"] == pytest.approx(expected_raw, **CLOSE)
    assert described.values["beta"] == (0.0 if target < 0.0 else 1.0)
    assert described.raw["beta"] != described.values["beta"] * MAX_GROSS_EXPOSURE
    assert not described.excluded
    assert _archive().insert(
        subject.agent_id, (0.0,), _ordered(descriptors, described.values)
    )


def _scaled_exposures(put_away, subject, scale: float):
    held = next(each for each in put_away.agents if each.agent_id == subject)
    return _with_agent(
        put_away,
        subject,
        exposures=tuple(scale * value for value in held.exposures),
    )


def _seed_turnover_above_one(descriptors, walked, put_away, monkeypatch):
    """A mandate that did not bind: the busiest agent's exposures scaled past the cap.

    Scaled so its in-window turnover lands at one and a half times ``c_t``.
    """
    busiest = max(
        _described_at_the_end(descriptors, walked),
        key=lambda each: each.raw["turnover"],
    )
    assert busiest.raw["turnover"] > 0.0, "nobody changed exposure in the window"
    subject = busiest.agent_id
    seeded = _scaled_exposures(put_away, subject, 1.5 / busiest.values["turnover"])
    held = next(each for each in seeded.agents if each.agent_id == subject)
    expected = oracle.turnover_raw(_tail(held.exposures)) / MAX_TURNOVER
    assert expected == pytest.approx(1.5)
    return seeded, subject, "turnover", expected


def _seed_horizon_above_one(descriptors, walked, put_away, monkeypatch):
    """A defective run-length computation: every mean run one longer than the window."""
    subject = _described_at_the_end(descriptors, walked)[0].agent_id
    monkeypatch.setattr(
        descriptors,
        "_mean_active_run_length",
        lambda active: active.shape[1] + 1.0 + 0.0 * active.sum(axis=1),
    )
    return put_away, subject, "holding_horizon", (WINDOW + 1.0) / WINDOW


def _seed_all_zero_reference(descriptors, walked, put_away, monkeypatch):
    """A reference that never moved: every retained reference return zero."""
    subject = _described_at_the_end(descriptors, walked)[0].agent_id
    seeded = replace(
        put_away, reference_returns=tuple(0.0 for _ in put_away.reference_returns)
    )
    return seeded, subject, "beta", math.nan


def _seed_an_exposure_not_finite(descriptors, walked, put_away, monkeypatch):
    """An exposure that is not a number, at the window's last decision point."""
    subject = _described_at_the_end(descriptors, walked)[0].agent_id
    held = next(each for each in put_away.agents if each.agent_id == subject)
    seeded = _with_agent(put_away, subject, exposures=(*held.exposures[:-1], math.nan))
    return seeded, subject, "turnover", math.nan


#: Each seed: what it seeds, the reason it is excluded for, and the window read.
SEEDS = {
    "turnover-above-one": (
        _seed_turnover_above_one,
        "outside_unit_interval",
        WINDOW,
    ),
    "horizon-above-one": (_seed_horizon_above_one, "outside_unit_interval", WINDOW),
    "all-zero-reference": (_seed_all_zero_reference, "undefined", WINDOW),
    "exposure-not-finite": (_seed_an_exposure_not_finite, "not_finite", WINDOW),
}


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_a_value_outside_the_unit_interval_is_excluded_never_clamped(
    descriptors: ModuleType, seed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value outside [0, 1], or not finite, reaches the record unaltered and is not binned.

    The value carried is exactly the declared computation's, never a bound in
    its place, it is marked excluded with its reason, and the platform archive
    refuses to bin it — the out-of-bounds raise that makes the exclusion
    visible where the adapter offers it.
    """
    walked, put_away = _finished(descriptors)
    seeding, reason, window = SEEDS[seed]
    seeded, subject, axis, expected = seeding(
        descriptors, walked, put_away, monkeypatch
    )

    described = descriptors.describe(
        agent.PopulationState.restored(seeded),
        _declaration(descriptors, window=window),
        subject,
    )

    assert described.values[axis] == pytest.approx(expected, nan_ok=True, **CLOSE)
    if math.isfinite(expected):
        assert not 0.0 <= described.values[axis] <= 1.0
    assert described.excluded == {axis: reason}
    if axis == "beta":
        assert math.isnan(described.raw["beta"])
    with pytest.raises(ValueError, match="outside bounds"):
        _archive().insert(subject, (0.0,), _ordered(descriptors, described.values))


#: The window the constant non-zero reference is read over: the configured
#: fitness window, where the computed mean of twenty returns of 0.01 is not
#: 0.01, so no mean taken of the reference can stand in for its move.
CONSTANT_REFERENCE_WINDOW = 20


def test_a_constant_nonzero_reference_is_described_not_excluded(
    descriptors: ModuleType,
) -> None:
    """A reference that moved by the same 0.01 at every decision point is a defined move.

    Seeded from a walked agent that earned a non-zero total over the window: its
    own outcomes, with every retained reference return replaced by 0.01. Beta through the origin is then the agent's mean
    outcome over 0.01, the oracle's value, and is described, not excluded.
    """
    walked, put_away = _finished(descriptors)
    held_by_id = {held.agent_id: held for held in put_away.agents}
    subject = next(
        each.agent_id
        for each in _described_at_the_end(descriptors, walked)
        if len(held_by_id[each.agent_id].history) >= CONSTANT_REFERENCE_WINDOW
        and math.fsum(held_by_id[each.agent_id].history[-CONSTANT_REFERENCE_WINDOW:])
        != 0.0
    )
    held = held_by_id[subject]
    seeded = replace(
        put_away, reference_returns=tuple(0.01 for _ in put_away.reference_returns)
    )

    described = descriptors.describe(
        agent.PopulationState.restored(seeded),
        _declaration(descriptors, window=CONSTANT_REFERENCE_WINDOW),
        subject,
    )

    expected_raw = oracle.beta_raw(
        held.history[-CONSTANT_REFERENCE_WINDOW:],
        (0.01,) * CONSTANT_REFERENCE_WINDOW,
    )
    assert math.isfinite(expected_raw) and expected_raw != 0.0
    assert described.raw["beta"] == pytest.approx(expected_raw, **CLOSE)
    assert described.values["beta"] == pytest.approx(
        oracle.beta_normalised(expected_raw, MAX_GROSS_EXPOSURE), **CLOSE
    )
    assert "beta" not in described.excluded


def test_an_agent_short_of_the_window_carries_no_descriptor(
    descriptors: ModuleType,
) -> None:
    """With ``W - 1`` outcomes an agent is undescribed and refused; with ``W`` it is described.

    The founders hold ``W - 1`` outcomes at decision point ``W - 1`` (none is
    recorded at the first decision point), so every one of them is refused
    there, by name, with what it held — and nothing is substituted: no value,
    no zero, no partial window.
    """
    windows, founding = _s1()
    readings: dict[int, object] = {}
    refusals: dict[str, object] = {}
    declaration = _declaration(descriptors)

    def read(index: int, state: agent.PopulationState) -> None:
        if index == WINDOW - 1:
            for agent_id in state.agent_ids():
                with pytest.raises(descriptors.DescriptorUnavailable) as refused:
                    descriptors.describe(state, declaration, agent_id)
                refusals[agent_id] = refused.value
        if index in (WINDOW - 1, WINDOW):
            readings[index] = descriptors.read(state, declaration)

    wired = _wired(founding)
    _at_each_turnover(wired, windows, read)
    wired.loop.run(  # ty: ignore[unresolved-attribute]
        streams.Watched(windows[: WINDOW + 1], wired.state)
    )

    founders = {birth.agent_id for birth in founding}
    short = readings[WINDOW - 1]
    surviving = founders & set(refusals)
    assert surviving, "every founder had died before the window could fill"
    assert short.described == {}
    assert set(short.undescribed) == set(refusals)
    for agent_id, refusal in refusals.items():
        assert isinstance(refusal, EcologyError)
        assert refusal.agent_id == agent_id
        assert refusal.held == short.undescribed[agent_id]
        assert refusal.held < WINDOW
        assert refusal.window == WINDOW
        assert agent_id in str(refusal)
    assert {short.undescribed[agent_id] for agent_id in surviving} == {WINDOW - 1}

    full = readings[WINDOW]
    assert surviving <= set(full.described)


def _comparable(reading) -> dict[str, tuple]:
    """A reading as values a ``==`` can compare, ``nan`` included."""
    return {
        agent_id: (
            tuple(float(value).hex() for value in described.values.values()),
            tuple(float(value).hex() for value in described.raw.values()),
            tuple(sorted(described.excluded.items())),
        )
        for agent_id, described in reading.described.items()
    } | {f"undescribed:{key}": (held,) for key, held in reading.undescribed.items()}


#: Where the walk is read and then carried on past: a trigger, well after the
#: warm-up, with most of the walk still to come.
READ_AT = 3 * 20 - 1


def test_a_descriptor_is_unchanged_by_later_decision_points(
    descriptors: ModuleType,
) -> None:
    """A reading taken at a decision point is the oracle's over what existed then, for good.

    Absolute, not shift-and-compare: the reading is held, the walk goes on to
    its end, and the held reading must still equal the oracle computed from
    histories truncated at its own instant. Then a replay whose bars after the
    reading are re-valued — S3's moves in place of S1's — must produce the same
    reading there.
    """
    windows, founding = _s1()
    walked = _walk(descriptors, windows, founding)
    held = walked.readings[READ_AT]
    assert held.described, "nobody was described at the reading"
    for agent_id, described in held.described.items():
        predicted = _predicted(walked, READ_AT, agent_id)
        assert predicted is not None
        _agrees(described, predicted)

    revalued = streams.priced(sign=-1, count=DECISION_POINTS)
    replay_windows = (*windows[: READ_AT + 1], *revalued[READ_AT + 1 :])
    assert replay_windows[READ_AT + 1].bars != windows[READ_AT + 1].bars
    replayed = _walk(descriptors, replay_windows, founding)

    assert _comparable(replayed.readings[READ_AT]) == _comparable(held)
    assert _comparable(replayed.readings[DECISION_POINTS - 1]) != _comparable(
        walked.readings[DECISION_POINTS - 1]
    ), "the re-valued bars changed nothing later, so the replay re-valued nothing"


def test_the_record_only_report_reads_the_populations_own_declaration(
    descriptors: ModuleType,
) -> None:
    """Every record-only trigger row reads the descriptors the population itself gives.

    A ``record_only`` walk over the declared streams
    (``tests/support/record_trace.py``): the population is built under a
    declaration and the loop handed that same declaration, as a composition
    does. **Mechanism: interception at the reading instant.** The bar source's
    ``look`` runs once the loop has finished a decision point and before the
    next is handed over — the instant the trigger's row reads its descriptors
    — and there asks the population port itself: ``descriptor(agent_id)`` for
    each living agent, and the reading that call is built on
    (``describe`` under the population's own declaration) for the raw beta the
    port's descriptor carries only clipped. Raw betas compare within
    :data:`CLOSE`: the row's reading is batched, the port's per agent, and the
    two may differ in the last places
    (``liq-docs/plans/oracle-v4-sufficiency-gaps-2026-10-14.md`` §2.1, §6).
    """
    from tests.support import record_trace

    own: dict[int, dict[str, tuple[object, float]]] = {}
    holder: dict[str, record_trace.Walked] = {}

    def look(index: int) -> None:
        if "walked" not in holder or index < 0:
            return
        population = holder["walked"].population
        described: dict[str, tuple[object, float]] = {}
        for agent_id in population.agent_ids():
            try:
                port = population.descriptor(agent_id)
            except descriptors.DescriptorUnavailable:
                continue
            reading = descriptors.describe(
                population.state, population.descriptors, agent_id
            )
            assert dict(port.values) == dict(reading.values), agent_id
            described[agent_id] = (port, reading.raw[descriptors.BETA])
        own[index] = described

    walked = record_trace.walk(
        diversity_record="record_only",
        look=look,
        before_run=lambda walked: holder.update(walked=walked),
    )
    rows = walked.run.report.triggers  # ty: ignore[unresolved-attribute]

    assert rows, "no trigger fell, so no row was read"
    assert any(row.described > 0 for row in rows), (
        "no row described anyone, so the declaration was never read"
    )
    assert any(row.raw_beta for row in rows), "no row carried a raw beta"
    for row in rows:
        expected = own[row.index]
        assert row.described == len(expected), row.index
        assert set(row.raw_beta) == set(expected), row.index
        for agent_id, raw in row.raw_beta.items():
            assert raw == pytest.approx(expected[agent_id][1], nan_ok=True, **CLOSE), (
                row.index,
                agent_id,
            )
