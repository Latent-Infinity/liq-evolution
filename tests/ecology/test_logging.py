"""What the boundaries this slice crosses write down, and what they cannot.

Three boundaries are instrumented: where bars become decision points, where a
mandate decides what may be held, and where a target is acted on. Two questions
are asked of each. Does the line say where it came from — which run, which
decision instant, which agent? And is there any route by which a raw payload, a
key or a credential could travel out with it?

The second question is answered structurally rather than by inspection. A test
that reads the lines this code emits today says nothing about the line somebody
adds tomorrow, so what is checked here is the *shape of the only writing
route*: one call, four parameters, none of them variadic, an event drawn from a
closed vocabulary, and no other logging machinery reachable from the three
modules at all. A value cannot be passed because there is nowhere to put it.

**Nothing here is market data.** The price paths below are declared arithmetic
in the sense the account's own checks use: they exist so that each boundary can
be crossed on purpose, and every number computed from them is a number about
the declared path.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import get_args

import pytest
import structlog

from liq.evolution.ecology import accounts as accounts_module
from liq.evolution.ecology import driver as driver_module
from liq.evolution.ecology import substrate as substrate_module
from liq.evolution.ecology.accounts import evaluate
from liq.evolution.ecology.config import EcologyConfig
from liq.evolution.ecology.driver import walk
from liq.evolution.ecology.features import FeatureDeclaration, FeatureSet
from liq.evolution.ecology.logctx import (
    EVENTS,
    LOGGABLE_KEYS,
    SILENT,
    Event,
    LineWriter,
    RunLog,
    redact_unlisted,
)
from liq.evolution.ecology.substrate import BarRecord, FeatureSubstrate
from liq.evolution.ecology.types import (
    AccountState,
    Bar,
    BarWindow,
    Fill,
    Intent,
    NotFilled,
    PositionTarget,
    Rejection,
    SizingOutcome,
)

#: Identity every run below is driven under.
RUN_ID = "logging-shape"
AGENT_ID = "logging-shape-agent"
INSTRUMENT = "DECLARED-A"

#: Where the declared paths start, and how long one of their bars lasts.
ORIGIN = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
BAR_DURATION = timedelta(minutes=1)

#: Vocabularies these checks declare. One feature is enough: what an agent
#: reads is the agent module's subject, not this one's.
LEVEL = "level"
SCHEMA_VERSION = "logging-shape-features-1"
DECLARED_SCENARIO = "logging-shape-scenario"

#: What an account opens with here. One, so an exposure and the equity it is a
#: fraction of are the same number.
OPENING_EQUITY = 1.0

#: A declared exposure that is neither flat nor the whole account.
HALF = 0.5

#: A declared price path, four bars long, so that a target formed at one
#: decision point and acted on at the next can be crossed more than once.
DECLARED_PATH = ((100.0, 101.0), (102.0, 104.0), (103.0, 99.0), (98.0, 100.0))

#: The modules this slice instruments. Every boundary record comes from one of
#: them, and the containment check below is asserted over exactly these.
INSTRUMENTED = (substrate_module, driver_module, accounts_module)

#: Logging machinery none of the three may reach. Reaching one would be a
#: second writing route, and a second route is one the shape checks here do not
#: cover.
OTHER_LOGGING = frozenset({"logging", "structlog", "loguru", "syslog"})


# --- a writer that keeps what it was handed ----------------------------------


@dataclass
class _Lines:
    """A line writer that keeps the context each line was bound with.

    Local rather than a logging library's capture helper, because what is under
    test is what :class:`RunLog` contributes to a line. Whether a library's own
    chain then renders it is that library's business and is checked where the
    redactor is.
    """

    written: list[tuple[Mapping[str, str], str]] = field(default_factory=list)
    context: Mapping[str, str] = field(default_factory=dict)

    def bind(self, **values: str) -> _Lines:
        """Return a writer carrying ``values`` beside what it already holds."""
        return _Lines(written=self.written, context={**self.context, **values})

    def info(self, event: str) -> None:
        """Keep one line: the context it was bound with, and its label."""
        self.written.append((dict(self.context), event))

    def events(self, *, stage: str | None = None) -> tuple[str, ...]:
        """The labels written, optionally only those from one step."""
        return tuple(
            event
            for context, event in self.written
            if stage is None or context["stage"] == stage
        )

    def contexts(self, event: str) -> tuple[Mapping[str, str], ...]:
        """The context of every line written under one label."""
        return tuple(context for context, written in self.written if written == event)


def _log() -> tuple[_Lines, RunLog]:
    """A writer and a log that writes into it."""
    lines = _Lines()
    return lines, RunLog(run_id=RUN_ID, writer=lines)


# --- declared streams and stand-ins ------------------------------------------


def _bar(index: int, opening: float, close: float) -> Bar:
    """One bar of the declared path."""
    period_start = ORIGIN + index * BAR_DURATION
    return Bar(
        instrument=INSTRUMENT,
        period_start=period_start,
        period_end=period_start + BAR_DURATION,
        open=opening,
        high=max(opening, close),
        low=min(opening, close),
        close=close,
        volume=1.0,
    )


@dataclass(frozen=True)
class _DeclaredPath:
    """Decision points over a declared price path, offered once and in order."""

    prices: tuple[tuple[float, float], ...] = DECLARED_PATH
    segment_ids: tuple[str, ...] = ()

    def windows(self) -> Iterator[BarWindow]:
        """Yield each decision point in chronological order, exactly once."""
        for index, (opening, close) in enumerate(self.prices):
            bar = _bar(index, opening, close)
            segment = (
                self.segment_ids[index] if self.segment_ids else "declared-segment"
            )
            yield BarWindow(
                as_of=bar.period_end,
                segment_id=segment,
                segment_role="train",
                bars=MappingProxyType({INSTRUMENT: bar}),
                features=MappingProxyType(
                    {INSTRUMENT: MappingProxyType({LEVEL: close})}
                ),
                feature_schema_version=SCHEMA_VERSION,
            )


@dataclass
class _WantsHalf:
    """A stand-in wanting the same declared exposure at every decision point."""

    agent_id: str = AGENT_ID

    def intend(self, window: BarWindow, instrument: str) -> Intent:
        """Want the declared exposure."""
        return Intent(
            agent_id=self.agent_id,
            instrument=instrument,
            target_exposure=HALF,
            as_of=window.as_of,
        )


#: The three shapes a well-formed sizing outcome has, scripted by name.
SIZED = "sized"
BOUND = "bound"
REFUSED = "refused"


@dataclass
class _ScriptedMandate:
    """A mandate answering each intent with a scripted, well-formed outcome."""

    shapes: tuple[str, ...]
    asked: int = 0

    def size(self, intent: Intent, account: AccountState) -> SizingOutcome:
        """Answer with whatever shape this decision point was scripted to have."""
        shape = self.shapes[self.asked]
        self.asked += 1
        target = PositionTarget(
            agent_id=intent.agent_id,
            instrument=intent.instrument,
            target_exposure=intent.target_exposure,
            as_of=intent.as_of,
        )
        rejection = Rejection(
            agent_id=intent.agent_id,
            instrument=intent.instrument,
            as_of=account.as_of,
            reason="declared_bound",
        )
        if shape == SIZED:
            return SizingOutcome(target=target, rejections=())
        if shape == BOUND:
            return SizingOutcome(target=target, rejections=(rejection,))
        return SizingOutcome(target=None, rejections=(rejection,))


@dataclass
class _ScriptedModel:
    """An execution model reaching, or not reaching, each target by script."""

    reached: tuple[bool, ...]
    acted: int = 0
    cost_scenario_id: str = DECLARED_SCENARIO

    def execute(
        self, target: PositionTarget, bar: Bar, account: AccountState
    ) -> tuple[Fill | NotFilled, AccountState]:
        """Reach the target, or report that it was not reached."""
        reached = self.reached[self.acted]
        self.acted += 1
        if not reached:
            return (
                NotFilled(
                    agent_id=target.agent_id,
                    instrument=target.instrument,
                    as_of=bar.period_start,
                    requested_exposure=target.target_exposure,
                    held_exposure=account.exposures.get(target.instrument, 0.0),
                    reason="declared_not_reached",
                    cost_scenario_id=self.cost_scenario_id,
                ),
                account,
            )
        fill = Fill(
            agent_id=target.agent_id,
            instrument=target.instrument,
            as_of=bar.period_start,
            requested_exposure=target.target_exposure,
            filled_exposure=target.target_exposure,
            price=bar.open,
            cost=0.0,
            cost_scenario_id=self.cost_scenario_id,
        )
        return fill, AccountState(
            agent_id=target.agent_id,
            as_of=bar.period_start,
            equity=account.equity,
            exposures=MappingProxyType({target.instrument: target.target_exposure}),
            costs_charged=account.costs_charged,
        )


def _evaluate(
    *,
    shapes: Sequence[str],
    reached: Sequence[bool],
    log: RunLog = SILENT,
):
    """Run one scripted agent over the declared path under ``log``."""
    return evaluate(
        _DeclaredPath(),
        EcologyConfig(run_id=RUN_ID),
        agent=_WantsHalf(),
        instrument=INSTRUMENT,
        sizer=_ScriptedMandate(shapes=tuple(shapes)),
        simulator=_ScriptedModel(reached=tuple(reached)),
        opening_equity=OPENING_EQUITY,
        log=log,
    )


def _substrate(*, closes: Sequence[float | None], log: RunLog) -> FeatureSubstrate:
    """A substrate over rows where a missing close is an incomplete bar."""
    records = []
    for index, close in enumerate(closes):
        period_start = ORIGIN + index * BAR_DURATION
        records.append(
            BarRecord(
                instrument=INSTRUMENT,
                period_start=period_start,
                period_end=period_start + BAR_DURATION,
                open=100.0,
                high=101.0,
                low=99.0,
                close=close,
                volume=1.0,
            )
        )
    return FeatureSubstrate(
        records=tuple(records),
        features=FeatureSet(
            declarations=(
                FeatureDeclaration(
                    name=LEVEL,
                    lookback=1,
                    availability_lag=0,
                    compute=lambda bars: bars[-1].close,
                ),
            ),
            maximum_lookback=1,
        ),
        feature_schema_version=SCHEMA_VERSION,
        log=log,
    )


# --- the writing route, and the fact that it is the only one -----------------


def test_the_event_vocabulary_is_closed_and_agrees_with_its_type() -> None:
    """The runtime tuple and the static type cannot drift apart unnoticed."""
    assert set(EVENTS) == set(get_args(Event.__value__))
    assert len(EVENTS) == len(set(EVENTS))


def test_a_line_has_no_parameter_a_value_could_be_passed_in() -> None:
    """The mechanism, stated: four parameters, none of them variadic.

    This is what makes "no raw payload, key or credential is loggable" a
    property of the code rather than a claim about today's call sites. A
    ``**values`` here would reopen the keyword route the context was built to
    close, and this check is what would fail when somebody adds one.
    """
    parameters = inspect.signature(RunLog.record).parameters
    assert list(parameters) == ["self", "event", "bar_ts", "stage", "agent_id"]
    kinds = {parameter.kind for parameter in parameters.values()}
    assert inspect.Parameter.VAR_KEYWORD not in kinds
    assert inspect.Parameter.VAR_POSITIONAL not in kinds


def test_a_value_offered_beside_a_line_is_refused_by_the_call_itself() -> None:
    """A credential has nowhere to go: the call does not accept one."""
    with pytest.raises(TypeError):
        SILENT.record(  # type: ignore[call-arg]
            "bar_excluded",
            bar_ts=ORIGIN,
            stage="feature",
            api_token="sk-live-DEADBEEF",
        )


def test_an_event_the_vocabulary_does_not_name_is_refused() -> None:
    """The label is a closed, code-chosen code, so nothing can be written into it.

    The refused string below is the shape a provider message really leaks in:
    a credential inside a URL, interpolated into what somebody thought was a
    message. It is not a message here, and there is no message to interpolate
    into.
    """
    with pytest.raises(ValueError, match="event must be one of"):
        SILENT.record(  # type: ignore[arg-type]
            "GET https://user:hunter2@feed.example/bars failed",
            bar_ts=ORIGIN,
            stage="feature",
        )


def test_a_silent_log_refuses_exactly_what_a_writing_one_refuses() -> None:
    """Silence is not permissiveness, so turning a log on cannot newly fail a run."""
    lines, log = _log()
    for candidate in (SILENT, log):
        with pytest.raises(ValueError, match="event must be one of"):
            candidate.record("not_an_event", bar_ts=ORIGIN, stage="feature")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="timezone-aware"):
            candidate.record(
                "bar_excluded", bar_ts=datetime(2024, 1, 2, 14, 30), stage="feature"
            )
    assert lines.written == []


def test_a_log_that_could_not_name_its_run_is_refused_where_it_is_built() -> None:
    """A line nothing can attribute to a run is refused before any is written.

    Refused at construction rather than at the first line, because a log built
    without an identity is a wiring mistake and every boundary that then wrote
    through it would report the same one.
    """
    lines = _Lines()
    with pytest.raises(ValueError, match="run_id"):
        RunLog(run_id="", writer=lines)


def test_a_silent_log_writes_nothing() -> None:
    """The default is instrumentation a run has not switched on, not a sink."""
    assert SILENT.record("bar_excluded", bar_ts=ORIGIN, stage="feature") is None


def test_a_written_line_carries_the_run_the_instant_and_the_agent() -> None:
    """The three keys the criteria name, on a line that concerns one agent."""
    lines, log = _log()
    log.record("intent_sized", bar_ts=ORIGIN, stage="size", agent_id=AGENT_ID)
    context, event = lines.written[0]
    assert event == "intent_sized"
    assert context["run_id"] == RUN_ID
    assert context["bar_ts"] == "2024-01-02T14:30:00+00:00"
    assert context["agent_id"] == AGENT_ID
    assert set(context) <= LOGGABLE_KEYS


def test_a_written_line_states_its_instant_in_utc() -> None:
    """A decision point at another offset is written down as the instant it is."""
    lines, log = _log()
    local = ORIGIN.astimezone(UTC).replace(tzinfo=UTC)
    log.record("decision_point_shown", bar_ts=local, stage="feature")
    assert lines.written[0][0]["bar_ts"].endswith("+00:00")


def test_a_real_bound_logger_is_a_line_writer() -> None:
    """The writer is named structurally, so no logging library is named inward."""
    sink = structlog.testing.CapturingLogger()
    logger = structlog.wrap_logger(
        sink,
        processors=[redact_unlisted, structlog.processors.KeyValueRenderer()],
    ).bind()
    assert isinstance(logger, LineWriter)

    RunLog(run_id=RUN_ID, writer=logger).record(
        "intent_refused", bar_ts=ORIGIN, stage="size", agent_id=AGENT_ID
    )
    line = sink.calls[0].args[0]
    assert "event='intent_refused'" in line
    assert f"run_id='{RUN_ID}'" in line
    assert f"agent_id='{AGENT_ID}'" in line
    assert "bar_ts='2024-01-02T14:30:00+00:00'" in line


def test_no_boundary_in_this_slice_writes_a_line_any_other_way() -> None:
    """One writing route, asserted rather than agreed.

    The shape checks above bound what can travel out through
    :meth:`RunLog.record`. They bound nothing at all if a module beside it
    holds a logger of its own, so the absence of a second route is checked
    here, over exactly the modules this slice instruments.
    """
    reached: dict[str, set[str]] = {}
    for module in INSTRUMENTED:
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        reached[module.__name__] = imported & OTHER_LOGGING
    assert reached == {module.__name__: set() for module in INSTRUMENTED}


# --- the bar boundary --------------------------------------------------------


def test_a_bar_that_was_never_complete_is_written_down_where_it_is_excluded() -> None:
    """The one thing only the substrate can say: a bar that produced nothing."""
    lines, log = _log()
    substrate = _substrate(closes=(100.0, None, 102.0), log=log)
    tuple(substrate.windows())

    assert lines.events() == ("bar_excluded",)
    context = lines.contexts("bar_excluded")[0]
    assert context["bar_ts"] == (ORIGIN + 2 * BAR_DURATION).isoformat()
    assert context["stage"] == "feature"


def test_a_complete_substrate_writes_no_exclusion() -> None:
    """A line is a boundary crossed, not a heartbeat."""
    lines, log = _log()
    tuple(_substrate(closes=(100.0, 101.0), log=log).windows())
    assert lines.written == []


def test_every_decision_point_the_walk_shows_is_written_down() -> None:
    """The bar boundary every source crosses, recorded where the walk crosses it."""
    lines, log = _log()
    walk(_DeclaredPath(), EcologyConfig(run_id=RUN_ID), log=log)

    shown = lines.contexts("decision_point_shown")
    assert len(shown) == len(DECLARED_PATH)
    assert [context["bar_ts"] for context in shown] == [
        (ORIGIN + (index + 1) * BAR_DURATION).isoformat()
        for index in range(len(DECLARED_PATH))
    ]
    assert {context["run_id"] for context in shown} == {RUN_ID}
    assert not [context for context in shown if "agent_id" in context]


def test_a_walk_forward_boundary_is_written_down_where_it_is_crossed() -> None:
    """A segment change is a boundary, and a boundary leaves a line."""
    lines, log = _log()
    walk(
        _DeclaredPath(segment_ids=("train", "train", "validate", "validate")),
        EcologyConfig(run_id=RUN_ID),
        log=log,
    )

    crossings = lines.contexts("segment_crossed")
    assert len(crossings) == 1
    assert crossings[0]["bar_ts"] == (ORIGIN + 3 * BAR_DURATION).isoformat()


# --- the risk and simulation boundaries --------------------------------------


def test_each_shape_the_mandate_can_decide_is_written_down_as_its_own_line() -> None:
    """Permitted, reduced and refused are three outcomes, not two."""
    lines, log = _log()
    _evaluate(shapes=(SIZED, BOUND, REFUSED, SIZED), reached=(True, False), log=log)
    assert lines.events(stage="size") == (
        "intent_sized",
        "intent_bound",
        "intent_refused",
        "intent_sized",
    )


def test_what_execution_reached_and_did_not_reach_are_written_down_apart() -> None:
    """A target that was not reached is as prominent as one that was."""
    lines, log = _log()
    _evaluate(shapes=(SIZED, BOUND, REFUSED, SIZED), reached=(True, False), log=log)
    assert lines.events(stage="fill") == ("target_reached", "target_not_reached")


def test_every_per_agent_line_names_the_agent_the_run_and_the_instant() -> None:
    """The acceptance criteria, over every line that concerns one agent."""
    lines, log = _log()
    _evaluate(shapes=(SIZED, BOUND, REFUSED, SIZED), reached=(True, False), log=log)
    per_agent = [
        context for context, _ in lines.written if context["stage"] in ("size", "fill")
    ]
    assert per_agent
    for context in per_agent:
        assert context["agent_id"] == AGENT_ID
        assert context["run_id"] == RUN_ID
        assert context["bar_ts"].endswith("+00:00")
        assert set(context) <= LOGGABLE_KEYS


def test_a_run_walked_under_a_log_scores_what_it_scores_unlogged() -> None:
    """Instrumentation is not an input: the same run produces the same account.

    Checked over the whole account rather than the score alone, because a
    figure can agree while the sequence behind it does not.
    """
    _, log = _log()
    logged = _evaluate(
        shapes=(SIZED, BOUND, REFUSED, SIZED), reached=(True, False), log=log
    )
    silent = _evaluate(shapes=(SIZED, BOUND, REFUSED, SIZED), reached=(True, False))

    assert logged.fitness == silent.fitness
    assert logged.states == silent.states
    assert logged.fills == silent.fills
    assert logged.rejections == silent.rejections
    assert logged.events == silent.events


def test_a_substrate_walked_under_a_log_yields_what_it_yields_unlogged() -> None:
    """The same property at the bar boundary, where the exclusions are decided."""
    _, log = _log()
    logged = _substrate(closes=(100.0, None, 102.0), log=log)
    silent = _substrate(closes=(100.0, None, 102.0), log=SILENT)

    assert [window.as_of for window in logged.windows()] == [
        window.as_of for window in silent.windows()
    ]
    assert logged.exclusions == silent.exclusions
