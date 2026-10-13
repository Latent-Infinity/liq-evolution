"""Evidence that parents are chosen on deflated figures only.

One claim. At a replacement trigger, each candidate is handed to selection with
the deflated figure that trigger's deflation returned for it, an agent the
deflation did not assess is handed with none, and the parents chosen are
unchanged when only any agent's undeflated figure changes.

The candidates at a trigger are the handed agents the deflation returned a
deflated figure for. The trigger is ordinary when they number at least the
population port's declared ``minimum_candidates``, read from the port the loop
holds: then each candidate is handed exactly its deflated figure and every other
agent none. Otherwise it is blind, and every agent is handed none.

An unassessed agent with no undeflated figure either cannot tell "handed its
undeflated figure" from "handed none", so the deflation here is a C-06 double
(``tests/support/reproduce_deflator.py``) that leaves a declared subset of
agents unassessed and gives every agent a real undeflated figure. The loop is
walked twice with the real population adapter over the declared stream of
``tests/support/turnover_stream.py``, the two runs differing only in the
double's undeflated figures: in one they sit above every deflated figure, in
the other below. Whoever is handed them, they would decide the tournament.

Nothing here is market data: the stream is declared arithmetic, and every
figure is the double's declaration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from liq.evolution import ecology
from liq.evolution.ecology.config import Inheritance
from liq.evolution.ecology.types import AgentId
from tests.support import reproduce_deflator as double
from tests.support import turnover_stream as streams

#: How many decision points each run walks.
DECISION_POINTS = 120


def _turnover():
    """The turnover both runs walk under; every key declared here."""
    return ecology.steady_state.TurnoverConfig(  # type: ignore[attr-defined]
        band_lower=24,
        band_upper=30,
        replacement_count=2,
        trigger_period=3,
        minimum_age=6,
        replacement_window=6,
        maximum_replaced_fraction=0.3,
        fitness_window=8,
        parents_per_birth=2,
        inheritance=Inheritance.BALDWINIAN,
        admission_redraws=8,
        seed=13,
    )


@dataclass(frozen=True)
class _Handed:
    """One call to selection: the trigger it followed and the figures it was handed."""

    trigger: int
    figures: dict[AgentId, float | None]


@dataclass(frozen=True)
class _Walked:
    """One run: what the double returned, what selection was handed and chose."""

    deflator: double.ReproduceDeflator
    handed: list[_Handed]
    selected: list[tuple[AgentId, ...]]
    births: tuple[tuple[AgentId, ...], ...]
    minimum_candidates: int


def _walk(undeflated: double.Undeflated) -> _Walked:
    """Walk the declared stream once under the double's ``undeflated`` figures.

    Selection is intercepted on the port the loop holds, so what is recorded is
    what the loop handed it, with the trigger — the double's latest call —
    whose deflation the figures should come from.
    """
    turnover = _turnover()
    deflator = double.ReproduceDeflator(undeflated=undeflated)
    wired = streams.wire(
        founding=streams.founders(turnover.band_lower),
        turnover=turnover,
        deflator=deflator,
    )
    handed: list[_Handed] = []
    choose = wired.port.select_parents

    def select(*, fitness, count, seed):
        handed.append(_Handed(trigger=len(deflator.calls) - 1, figures=dict(fitness)))
        return choose(fitness=fitness, count=count, seed=seed)

    wired.port.select_parents = select  # type: ignore[method-assign]
    run = wired.loop.run(  # type: ignore[attr-defined]
        streams.Watched(streams.priced(count=DECISION_POINTS), wired.state)
    )
    return _Walked(
        deflator=deflator,
        handed=handed,
        selected=list(wired.port.selected),
        births=tuple(birth.parents for birth in run.report.births),
        minimum_candidates=wired.port.minimum_candidates,
    )


def _candidates(walked: _Walked, call: _Handed) -> dict[AgentId, float]:
    """The handed agents the trigger's deflation returned a deflated figure for."""
    return {
        verdict.agent_id: verdict.deflated
        for verdict in walked.deflator.calls[call.trigger]
        if verdict.agent_id in call.figures and verdict.deflated is not None
    }


def _ordinary(walked: _Walked, call: _Handed) -> bool:
    """Whether the trigger has at least the port's declared number of candidates."""
    return len(_candidates(walked, call)) >= walked.minimum_candidates


def _premise_problems(walked: _Walked, label: str) -> list[str]:
    """What leaves a run unable to tell a deflated figure from an undeflated one."""
    problems: list[str] = []
    if not walked.births:
        problems.append(f"{label}: no birth")
    asked = [
        call
        for call in walked.handed
        if any(figure is not None for figure in call.figures.values())
        and any(
            verdict.agent_id in call.figures
            and verdict.deflated is None
            and verdict.undeflated is not None
            for verdict in walked.deflator.calls[call.trigger]
        )
    ]
    if not asked:
        problems.append(
            f"{label}: no trigger on figures was handed an unassessed agent "
            "carrying an undeflated figure"
        )
    if not any(
        _ordinary(walked, call)
        and any(
            call.figures[agent_id] is not None for agent_id in _candidates(walked, call)
        )
        for call in walked.handed
    ):
        problems.append(
            f"{label}: no ordinary trigger at which a candidate was handed a figure"
        )
    return problems


def _handed_problems(walked: _Walked, label: str) -> list[str]:
    """Every agent handed to selection with other than the figure the rule names.

    At an ordinary trigger a candidate is owed exactly its deflated figure and
    every other agent none; at a blind trigger every agent is owed none.
    """
    problems: list[str] = []
    for call in walked.handed:
        candidates = _candidates(walked, call)
        ordinary = len(candidates) >= walked.minimum_candidates
        kind = "ordinary" if ordinary else "blind"
        for agent_id, figure in sorted(call.figures.items()):
            owed = candidates.get(agent_id) if ordinary else None
            if figure is None and owed is None:
                continue
            if figure != owed:
                problems.append(
                    f"{label}: at {kind} trigger {call.trigger} {agent_id} was "
                    f"handed {figure!r}; it is owed {owed!r}"
                )
    return problems


def _trigger_counts(walked: _Walked) -> tuple[int, int]:
    """How many selections followed an ordinary trigger, and how many a blind one."""
    ordinary = sum(_ordinary(walked, call) for call in walked.handed)
    return ordinary, len(walked.handed) - ordinary


def test_parents_are_chosen_on_deflated_figures_only(
    record_property: Callable[[str, object], None],
) -> None:
    """Selection is handed exactly the figures the rule names; parents ignore undeflated ones.

    Two runs, identical but for every agent's undeflated figure. In each, at
    every selection, the candidates are the handed agents for which the
    double's deflation at that trigger returned a deflated figure. If they
    number at least the port's declared ``minimum_candidates`` (an ordinary
    trigger), each candidate is handed exactly that figure and every other
    agent ``None``; otherwise (a blind trigger) every agent is handed ``None``.
    The parents chosen at every trigger, and each birth's parents, are the same
    in both runs. The number of ordinary and blind triggers in each run is
    recorded.
    """
    walks = {
        "high undeflated": _walk(double.HIGH_UNDEFLATED),
        "low undeflated": _walk(double.LOW_UNDEFLATED),
    }
    counts = {label: _trigger_counts(walked) for label, walked in walks.items()}
    seen = "; ".join(
        f"{label}: {ordinary} ordinary and {blind} blind triggers"
        for label, (ordinary, blind) in counts.items()
    )
    for label, (ordinary, blind) in counts.items():
        record_property(f"{label} ordinary triggers", ordinary)
        record_property(f"{label} blind triggers", blind)
    premises = [
        problem
        for label, walked in walks.items()
        for problem in _premise_problems(walked, label)
    ]
    assert premises == [], f"{premises} ({seen})"

    handed = [
        problem
        for label, walked in walks.items()
        for problem in _handed_problems(walked, label)
    ]
    assert handed == [], (
        f"{handed[0]}; {len(handed)} agents handed to selection with other than "
        f"the figure the rule names ({seen})"
    )

    high, low = walks.values()
    differing = [
        trigger
        for trigger, (first, second) in enumerate(
            zip(high.selected, low.selected, strict=True)
        )
        if first != second
    ]
    assert differing == [] and high.births == low.births, (
        f"the parents chosen differ at {len(differing)} of {len(high.selected)} "
        "selections when only undeflated figures differ"
    )
