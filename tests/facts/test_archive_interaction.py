"""Evidence that the loop offers its diversity record exactly as declared, and nothing more.

``ECOLOGY.NICHE.ARCHIVE_INTERACTION_DECLARED.v1`` (EV-E79): given a steady-state
run, when a replacement trigger's eligible agents have been deflated, then with
the record's offers configured on (``diversity_record = "record_only"``) the loop
— and nothing else — offers the record one entry for each of them that the
trigger's deflation assessed and whose descriptor is defined, before the
trigger's first death and at no other time, each carrying as its only figure,
under ``"deflated"``, exactly the figure that deflation returned for it; with
the offers off, the default, nothing is offered; no agent without a deflated
figure is ever offered; nothing the loop decides reads the record, so discarding
every offer changes nothing; the record carries across split boundaries and
every run starts with it empty; and an eviction never removes a living agent,
nor a death an entry.

**Observation, not self-report.** What was offered is read from the real
adapter's ``record`` and the platform archive's ``insert``, intercepted at the
instance by calling module, against the C-06 double's call log
(``tests/support/record_trace.py``). The evidence shape is
``liq-docs/plans/oracle-v4-archive-interaction-2026-10-02.md`` §6, under
``liq-docs/plans/oracle-v4-offers-under-no-rule-2026-10-13.md``. Configuring the
offers on is not niching: no niching rule acts on the living, and node names
that say "niching" are labels.

**The default run's reference** is a frozen fixture, the sha256 of the
turnover report the loop made at the same seed before it could offer anything (:data:`record_trace.OFFER_FREE_REPORT_DIGEST`).

**Seeded offers.** A founder offered with 0.0 at admission, an unassessed agent
offered with its undeflated figure, and a constant figure at every trigger are
each required to turn the trace red, so the trace cannot pass by being unable
to fail.

Nothing here is market data: every stream is declared arithmetic.
"""

from __future__ import annotations

import math

import pytest

from liq.evolution import ecology
from liq.evolution.ecology import descriptors
from liq.evolution.ecology.types import ArchiveEntry, Descriptor
from tests.support import level_shift_stream as s1
from tests.support import record_trace
from tests.support import turnover_stream as streams

RECORD_ONLY = "record_only"


def _offered(**keys: object) -> record_trace.Walked:
    return record_trace.walk(diversity_record=RECORD_ONLY, **keys)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("sign", [1, -1], ids=["S1", "S3"])
def test_only_the_loop_offers_each_assessed_described_agent_before_the_first_death(
    sign: int,
) -> None:
    walked = _offered(windows=streams.priced(sign, count=record_trace.DECISION_POINTS))
    trace = walked.trace
    report = walked.run.report  # ty: ignore[unresolved-attribute]

    assert record_trace.offer_problems(walked) == []
    assert trace.offers, "nothing was offered, so the trace was never exercised"
    assert any(not offer.kept for offer in trace.offers), (
        "every offer was kept, so the record never compared one"
    )
    assessed = sum(
        verdict.deflated is not None
        for call in walked.deflator.calls
        for verdict in call
    )
    assert assessed > len(trace.offers), (
        "every assessed agent was described, so the undefined-descriptor rule "
        "was never exercised"
    )
    assert any(kind == "death" for kind, _ in trace.events)
    undeflated = {
        (bar, verdict.agent_id): verdict.undeflated
        for bar, call in zip(
            (bar for kind, bar in trace.events if kind == "deflate"),
            walked.deflator.calls,
            strict=True,
        )
        for verdict in call
    }
    assert any(
        offer.entry.objectives["deflated"]
        != undeflated[(offer.bar, offer.entry.agent_id)]
        for offer in trace.offers
    ), "every offered figure equalled the undeflated one, so they were never told apart"
    by_bar: dict[int, list[record_trace.Offer]] = {}
    for offer in trace.offers:
        by_bar.setdefault(offer.bar, []).append(offer)
    for row in report.triggers:
        made = by_bar.get(row.index, [])
        assert row.offers == len(made), row.index
        assert row.offers_kept == sum(offer.kept for offer in made), row.index
        assert row.not_offered_descriptor_undefined == (
            row.indistinguishable + row.worse + row.better - row.offers
        ), row.index


def test_discarding_every_offer_changes_nothing_the_loop_decides() -> None:
    """The same seed, every offer kept or every offer discarded: the same run."""
    kept = _offered()
    discarded = _offered(discard_offers=True)

    assert kept.population.recorded(), "nothing was kept, so nothing was discarded"
    assert discarded.population.recorded() == ()
    assert record_trace.decided(kept) == record_trace.decided(discarded)


def test_niching_off_offers_nothing() -> None:
    """The default offers nothing, and its report is the offer-free loop's to the byte."""
    walked = record_trace.reference_walk()
    report = walked.run.report  # ty: ignore[unresolved-attribute]

    assert report.births, "nobody was born, so no birth could have offered"
    assert {death.kind for death in report.deaths} >= {"cull", "retirement"}, (
        "a kind of death this stream yields never happened, so it was never tested"
    )
    assert walked.trace.offers == [], "the default run offered to the record"
    assert walked.trace.inserts == []
    assert walked.population.recorded() == ()
    assert type(report) is ecology.steady_state.TurnoverReport
    assert (
        record_trace.report_digest(report) == record_trace.OFFER_FREE_REPORT_DIGEST
    ), (
        "the default run's turnover report is not byte-identical to the offer-free loop's"
    )


def test_the_record_carries_across_a_split_and_a_run_starts_it_empty() -> None:
    """Across the boundary nothing is reset or re-binned; a new run starts empty."""
    cells: dict[int, set[tuple[int, ...]]] = {}
    bins: dict[int, tuple[int, ...]] = {}
    holder: dict[str, record_trace.Walked] = {}

    def look(index: int) -> None:
        if index in (s1.BREAK_AT - 1, s1.BREAK_AT):
            archive = holder["walked"].population.archive
            cells[index] = {tuple(raw["bin"]) for raw in archive.to_dict()["entries"]}  # ty: ignore[not-iterable]
            bins[index] = tuple(archive.bins_per_dim)

    def started(walked: record_trace.Walked) -> None:
        holder["walked"] = walked
        assert walked.population.recorded() == (), "a run began with a record"

    windows = streams.priced(count=s1.BREAK_AT + 30)
    walked = _offered(
        windows=windows,
        record_carries_across_segments=True,
        look=look,
        before_run=started,
    )
    rows = walked.run.report.triggers  # ty: ignore[unresolved-attribute]

    assert cells[s1.BREAK_AT - 1], "the record was empty at the boundary"
    assert cells[s1.BREAK_AT - 1] <= cells[s1.BREAK_AT], "the boundary emptied a cell"
    assert bins[s1.BREAK_AT - 1] == bins[s1.BREAK_AT]
    assert len({row.segment_id for row in rows}) == 2, "no trigger fell on each side"
    filled = [row.cumulative_coverage.filled for row in rows]
    assert filled == sorted(filled), "cumulative coverage fell across the run"

    again = _offered(windows=windows, before_run=started)
    first = next(row for row in rows if row.offers)
    first_again = next(row for row in again.run.report.triggers if row.offers)  # ty: ignore[unresolved-attribute]
    assert first_again.cumulative_coverage == first.cumulative_coverage, (
        "a second run's first offers found the first run's entries"
    )


def test_eviction_and_death_stay_independent_under_niching() -> None:
    """An eviction removes no living agent; a death removes no entry."""
    walked = _offered()
    trace = walked.trace

    evicted_living = 0
    for offer in trace.offers:
        assert offer.living_after == offer.living_before, (
            f"offering {offer.entry.agent_id} changed who is alive"
        )
        gone = set(map(id, offer.held_before)) - set(map(id, offer.held_after))
        evicted_living += sum(
            id(entry) in gone and entry.agent_id in offer.living_before
            for entry in offer.held_before
        )
    with_entries = 0
    for _, agent_id, before, after in trace.deaths:
        assert before == after, f"the death of {agent_id} changed the record"
        with_entries += any(entry.agent_id == agent_id for entry in before)
    assert evicted_living, (
        "no living agent's entry was evicted, so eviction was untested"
    )
    assert with_entries, "no agent with an entry died, so death was untested"

    # Across each trigger's deaths as a whole: the record from just before its
    # first death to once the loop is done with the trigger, before any later
    # offer, so a removal after a departure returns is seen too.
    first_death: dict[int, tuple[ArchiveEntry, ...]] = {}
    died_with_entries: dict[int, int] = {}
    for bar, agent_id, before, _ in trace.deaths:
        first_death.setdefault(bar, before)
        died_with_entries[bar] = died_with_entries.get(bar, 0) + any(
            entry.agent_id == agent_id for entry in first_death[bar]
        )
    assert len(trace.held_after_bar) == len(trace.bars)
    for bar, before in first_death.items():
        assert trace.held_after_bar[bar] == before, (
            f"the deaths at bar {bar} changed the record"
        )
    assert any(died_with_entries.values()), (
        "no trigger's deaths took an agent holding an entry, so the comparison "
        "across a trigger's deaths was untested"
    )


def _window_sum(sample) -> float:
    """The undeflated figure the double hands an agent it leaves unassessed."""
    return math.fsum(sample.outcomes)


def test_an_unassessed_agent_holding_an_undeflated_figure_is_never_offered() -> None:
    """An agent the deflation left unassessed is not offered, whatever figure it carries.

    The record-only walk over ``tests/support/turnover_stream.py`` with a C-06
    double that leaves a declared subset of the eligible unassessed
    (``deflated is None``: every agent whose name ends in 0 or 5) while handing
    each of them a real, non-``None`` undeflated figure, its window's sum. The
    double is ``tests/support/record_trace.py``'s ``TracedDeflator``, extended
    for this case with ``unassessed_undeflated``; ``reproduce_deflator.py`` is
    not used. Without such a figure "offered its undeflated figure" and
    "offered nothing" are the same ``None``
    (``liq-docs/plans/oracle-v4-sufficiency-gaps-2026-10-14.md`` §2.1).
    """
    walked = _offered(
        verdicts=streams.per_sample(_silent_on_some), unassessed_undeflated=_window_sum
    )
    trace = walked.trace
    calls = dict(
        zip(
            (bar for kind, bar in trace.events if kind == "deflate"),
            walked.deflator.calls,
            strict=True,
        )
    )

    carrying = sum(len(agents) for agents in trace.unassessed_carrying.values())
    assert any(trace.unassessed_carrying.values()), (
        "no trigger held an unassessed, described agent carrying an undeflated "
        "figure, so the clause was never exercised"
    )
    assert carrying > 0
    assert trace.offers, "nothing was offered, so the trace was never exercised"
    for offer in trace.offers:
        verdict = {each.agent_id: each for each in calls[offer.bar]}.get(
            offer.entry.agent_id
        )
        assert verdict is not None and verdict.deflated is not None, (
            f"bar {offer.bar}: {offer.entry.agent_id} was offered without being assessed"
        )
        assert dict(offer.entry.objectives) == {"deflated": verdict.deflated}, (
            f"bar {offer.bar}: {offer.entry.agent_id} was offered another figure"
        )
    assert record_trace.offer_problems(walked) == []


def _founder_offered_zero(monkeypatch: pytest.MonkeyPatch) -> dict:
    built = ecology.steady_state.SteadyState.__init__
    at = streams.priced(count=1)[0].as_of

    def offering_founders(self, **keys):
        built(self, **keys)
        for agent_id in self.population.agent_ids():
            self.population.record(
                ArchiveEntry(
                    agent_id=agent_id,
                    genome=self.population.genome(agent_id),
                    descriptor=Descriptor(
                        values=dict.fromkeys(descriptors.AXES, 0.5),
                        schema_version=descriptors.VOCABULARY,
                    ),
                    objectives={"deflated": 0.0},
                    recorded_at=at,
                )
            )

    monkeypatch.setattr(ecology.steady_state.SteadyState, "__init__", offering_founders)
    return {}


def _silent_on_some(sample):
    """The double declines every agent whose name ends in 0 or 5."""
    if sample.agent_id.endswith(("0", "5")):
        return None
    return streams.negative_is_worse(sample)


def _unassessed_offered_undeflated(monkeypatch: pytest.MonkeyPatch) -> dict:
    offer = ecology.steady_state.SteadyState._offer  # ty: ignore[unresolved-attribute]

    def also_the_unassessed(self, window, verdicts):
        made = offer(self, window, verdicts)
        for verdict in verdicts:
            if verdict.deflated is not None:
                continue
            try:
                described = self.population.descriptor(verdict.agent_id)
            except descriptors.DescriptorUnavailable:
                continue
            window_outcomes = self.state.history(verdict.agent_id)
            self.population.record(
                ArchiveEntry(
                    agent_id=verdict.agent_id,
                    genome=self.population.genome(verdict.agent_id),
                    descriptor=described,
                    objectives={
                        "deflated": math.fsum(
                            window_outcomes[-self.turnover.fitness_window :]
                        )
                    },
                    recorded_at=window.as_of,
                )
            )
        return made

    monkeypatch.setattr(ecology.steady_state.SteadyState, "_offer", also_the_unassessed)
    return {"verdicts": streams.per_sample(_silent_on_some)}


def _constant_figure(monkeypatch: pytest.MonkeyPatch) -> dict:
    offer = ecology.steady_state.SteadyState._offer  # ty: ignore[unresolved-attribute]

    def constant(self, window, verdicts):
        return offer(self, window, record_trace.with_figures(verdicts, lambda _: 1.0))

    monkeypatch.setattr(ecology.steady_state.SteadyState, "_offer", constant)
    return {}


@pytest.mark.parametrize(
    "seed",
    [_founder_offered_zero, _unassessed_offered_undeflated, _constant_figure],
    ids=["founder-zero", "unassessed-undeflated", "constant-figure"],
)
def test_a_forbidden_offer_turns_the_trace_red(seed, monkeypatch) -> None:
    """Each placeholder offer the records forbid is caught by the trace."""
    record_trace.ecology_config(
        run_id="seeded", turnover=record_trace.turnover(), diversity_record=RECORD_ONLY
    )
    keys = seed(monkeypatch)
    walked = _offered(**keys)
    assert record_trace.offer_problems(walked) != []
