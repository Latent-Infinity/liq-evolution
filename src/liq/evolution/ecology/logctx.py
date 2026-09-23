"""What every ecology log line says about where it came from, and nothing else.

Four keys locate a line: which run emitted it (``run_id``), which decision
point it belongs to (``bar_ts``), which agent it concerns (``agent_id``, absent
where the line is about the whole population rather than one member) and which
step of the per-bar pipeline produced it (``stage``). Together they let two
lines from different runs, agents or steps be told apart without reading their
text, which is the whole reason the context is structured rather than
formatted into a message.

``stage`` names a step of the forward pass — ``feature``, ``size``, ``fill``,
``learn``, ``allocate``. The vocabulary is closed and checked at construction,
so a misspelling is a failure rather than a value that quietly never matches a
filter.

``bar_ts`` is UTC. A naive timestamp and a timestamp at any other offset are
both refused when the context is built, so a line carrying a local time cannot
be emitted at all; conversion to a local zone belongs at a display boundary.

**Why an allow-list.** :func:`redact_unlisted` keeps the value of a key named
in :data:`LOGGABLE_KEYS` and replaces every other value with
:data:`REDACTED`. A deny-list of forbidden names — ``token``, ``password``,
``api_key`` — only refuses what someone thought of in advance, and the field
that leaks a credential is by definition the one nobody anticipated: a
provider's ``headers`` mapping, a ``request`` object whose repr embeds a signed
URL, a ``row`` carrying a raw payload. An allow-list fails closed on all of
them. The cost is real and accepted: a slice that wants a genuinely new key
must add it here, where the addition is visible and reviewable, instead of
passing it as a keyword argument nobody sees again.

The message itself — structlog's ``event`` — is permitted, and it is the one
place a value can still escape. Event names are static, code-chosen labels
such as ``"intent_refused"``; a value belongs in a key, never interpolated into
the label.

**Why there is a writing route here and not in each boundary.**
:class:`RunLog` is the one call a boundary in this package writes a line
through, and it takes four arguments: a label from the closed vocabulary
:data:`EVENTS`, the decision instant, the step, and the agent where the line
concerns one. None of them is variadic, so there is no parameter a payload, a
key or a credential could be passed in — the keyword route
:func:`redact_unlisted` redacts is not merely redacted here, it is absent. That
is the difference between a rule callers follow and a rule they cannot break,
and it is why the boundaries below do not each hold a logger of their own.

The closed label vocabulary closes the remaining route. Interpolating a value
into a message is how a credential inside a provider's URL reaches a log, and a
label outside :data:`EVENTS` is refused rather than written: an interpolated
string is never one of the eight names. The refusal is a ``ValueError`` where
:func:`redact_unlisted` raises nothing, and the two differ because they judge
different things — the redactor judges values that vary with the tape, where
failing a run over a log line would be absurd, while a label and a step are
code constants whose correctness does not depend on any bar.

:data:`SILENT` is what a boundary writes through when a run has wired no
writer. It discards every line and refuses everything :class:`RunLog` refuses,
so switching a writer on cannot make a run newly fail at a call that was
already being made.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from datetime import UTC, timedelta
from typing import Literal, Protocol, Self, runtime_checkable

from liq.evolution.ecology.types import AgentId, UtcTimestamp

__all__ = [
    "EVENTS",
    "LOGGABLE_KEYS",
    "REDACTED",
    "SILENT",
    "STAGES",
    "UNWIRED",
    "ContextBinder",
    "Event",
    "LineWriter",
    "LogContext",
    "RunLog",
    "Stage",
    "redact_unlisted",
]

# A step of the per-bar forward pass. Closed: the five steps are the pipeline.
type Stage = Literal["feature", "size", "fill", "learn", "allocate"]

# The same vocabulary at runtime, so a value can be checked and not only typed.
STAGES: tuple[Stage, ...] = ("feature", "size", "fill", "learn", "allocate")

# What a boundary in this package can say happened. Closed, and closed for a
# reason beyond tidiness: a label outside this set is refused, so a value
# interpolated into what somebody took for a message is never written.
type Event = Literal[
    "bar_excluded",
    "decision_point_shown",
    "intent_bound",
    "intent_refused",
    "intent_sized",
    "segment_crossed",
    "target_not_reached",
    "target_reached",
]

# The same vocabulary at runtime, for the reason :data:`STAGES` is.
EVENTS: tuple[Event, ...] = (
    "bar_excluded",
    "decision_point_shown",
    "intent_bound",
    "intent_refused",
    "intent_sized",
    "segment_crossed",
    "target_not_reached",
    "target_reached",
)

# The only keys whose values survive :func:`redact_unlisted`.
LOGGABLE_KEYS: frozenset[str] = frozenset({"agent_id", "bar_ts", "run_id", "stage"})

# What a key outside the allow-list is worth once it reaches the log.
REDACTED = "[redacted]"

# ``event`` is structlog's name for the static label of the line itself.
_PERMITTED_KEYS: frozenset[str] = LOGGABLE_KEYS | {"event"}


@runtime_checkable
class ContextBinder(Protocol):
    """Anything that carries key/value context forward, such as a bound logger.

    Named structurally so that :meth:`LogContext.bind` neither imports a
    logging library into the domain nor ties the ecology to one.
    """

    def bind(self, **values: str) -> Self:
        """Return a binder carrying ``values`` in addition to what it holds."""
        ...


@runtime_checkable
class LineWriter(ContextBinder, Protocol):
    """A binder that can also write the line it has been carrying context for.

    Two methods rather than one, and structural for the reason
    :class:`ContextBinder` is: a bound logger from any library satisfies it,
    and none of them is named inside the domain.
    """

    def info(self, event: str, /) -> object:
        """Write one line labelled ``event``, carrying whatever is bound."""
        ...


@dataclass(frozen=True)
class LogContext:
    """Where one log line came from.

    The dataclass is the first half of the allow-list: it has exactly these
    fields, so a credential, a key or a raw payload has nowhere to be put. The
    second half is :func:`redact_unlisted`, which closes the keyword-argument
    route the logging library leaves open.

    Attributes:
        run_id: Identity of the run the line was emitted by.
        bar_ts: The decision point the line belongs to (UTC). A naive or
            non-UTC value is refused when the context is built.
        stage: Which step of the per-bar forward pass produced the line.
        agent_id: The agent the line concerns, or ``None`` where it concerns
            the whole population — computing the shared feature view, or
            allocating across agents, belongs to no single member.

    Raises:
        ValueError: If ``run_id`` is empty, ``stage`` is outside the closed
            vocabulary, ``agent_id`` is given but empty, or ``bar_ts`` is naive
            or at an offset other than UTC.
    """

    run_id: str
    bar_ts: UtcTimestamp
    stage: Stage
    agent_id: AgentId | None = None

    def __post_init__(self) -> None:
        """Refuse a context that could not locate the line it labels."""
        if not self.run_id:
            raise ValueError("run_id must name the run that emitted the line")
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, not {self.stage!r}")
        if self.agent_id is not None and not self.agent_id:
            raise ValueError("agent_id, where given, must name an agent")
        offset = self.bar_ts.utcoffset()
        if offset is None:
            raise ValueError("bar_ts must be timezone-aware; a naive value is not UTC")
        if offset != timedelta(0):
            raise ValueError(
                f"bar_ts must be UTC, not {self.bar_ts.strftime('%z')}; "
                "convert to a local zone for display only"
            )

    def as_event_dict(self) -> Mapping[str, str]:
        """Return exactly the keys this context contributes to a line.

        ``bar_ts`` is rendered as an ISO-8601 instant at ``+00:00``, so the
        offset is visible in the output rather than implied by the type.
        ``agent_id`` is absent rather than null where the line concerns the
        whole population, because a key that is present and empty reads as a
        missing value instead of an inapplicable one.
        """
        context = {
            "bar_ts": self.bar_ts.astimezone(UTC).isoformat(),
            "run_id": self.run_id,
            "stage": self.stage,
        }
        if self.agent_id is not None:
            context["agent_id"] = self.agent_id
        return context

    def bind[BinderT: ContextBinder](self, logger: BinderT) -> BinderT:
        """Return ``logger`` carrying this context on every line it emits."""
        return logger.bind(**self.as_event_dict())


def redact_unlisted(
    _logger: object,
    _method_name: str,
    event_dict: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    """Replace the value of every key outside the allow-list with a marker.

    A structlog processor. It must be the **first** entry in the chain: it
    judges what the caller supplied, and anything a later processor adds — the
    level, a timestamp, the logger's name — is added after it has run and is
    therefore untouched. Placed later it would redact those additions too,
    which is loud and obvious rather than silent, but is still the wrong order.

    The key is kept and only its value replaced, so a line makes it visible
    that something was offered and refused. Nothing is raised: a log call is
    not a place to fail a run.
    """
    return {
        key: value if key in _PERMITTED_KEYS else REDACTED
        for key, value in event_dict.items()
    }


@dataclass(frozen=True)
class _Discarding:
    """A writer that keeps nothing, so that a log can be off without being absent.

    The alternative — an optional writer and a branch at every boundary — makes
    "does this run log?" a question each call site answers for itself, and one
    call site answering it differently is exactly the kind of difference that
    goes unnoticed. Here every boundary calls unconditionally and this is what
    the call reaches when nothing was wired.
    """

    def bind(self, **_values: str) -> Self:
        """Carry nothing forward; there is nowhere for it to go."""
        return self

    def info(self, _event: str, /) -> None:
        """Keep nothing."""
        return None


#: Identity a log no run wired writes under. Nothing is ever written under it —
#: :data:`SILENT` discards every line — and it exists so the refusals below hold
#: identically whether or not a writer was wired. A run that wires one supplies
#: its own identity with it.
UNWIRED = "unwired"


@dataclass(frozen=True)
class RunLog:
    """The one way a boundary in this package writes a line.

    One call, four arguments, none of them variadic. What a line can carry is
    therefore the four context keys and a label from a closed vocabulary, and a
    payload, a key or a credential is not refused at the boundary so much as
    unable to reach it: there is no parameter to pass one in.

    Attributes:
        run_id: Identity of the run every line written through this log names.
            Held here rather than passed per line because a boundary that had
            to supply it would need the run's configuration to log, and the
            substrate is a boundary that does not have one.
        writer: What the line is handed to once the context is bound to it.

    Raises:
        ValueError: If ``run_id`` is empty.
    """

    run_id: str
    writer: LineWriter

    def __post_init__(self) -> None:
        """Refuse a log whose lines could not say which run wrote them."""
        if not self.run_id:
            raise ValueError("run_id must name the run whose boundaries are recorded")

    def record(
        self,
        event: Event,
        *,
        bar_ts: UtcTimestamp,
        stage: Stage,
        agent_id: AgentId | None = None,
    ) -> None:
        """Write one line saying ``event`` happened, and where.

        Args:
            event: What happened, from :data:`EVENTS`. A label outside the
                vocabulary is refused rather than written, so a value cannot
                arrive interpolated into one.
            bar_ts: The decision instant the line belongs to (UTC).
            stage: Which step of the per-bar forward pass crossed the boundary.
            agent_id: The agent the line concerns, or ``None`` where it
                concerns the whole population.

        Raises:
            ValueError: If ``event`` is outside :data:`EVENTS`, or if the
                context the line would carry is not one that locates it —
                ``stage`` outside its vocabulary, ``bar_ts`` naive or at
                another offset, ``agent_id`` given but empty.
        """
        if event not in EVENTS:
            raise ValueError(f"event must be one of {EVENTS}, not {event!r}")
        context = LogContext(
            run_id=self.run_id, bar_ts=bar_ts, stage=stage, agent_id=agent_id
        )
        context.bind(self.writer).info(event)


#: What a boundary writes through until a run wires a writer. Silent, and as
#: strict as a writing log: a malformed record fails where it is made rather
#: than the first time somebody switches logging on.
SILENT: RunLog = RunLog(run_id=UNWIRED, writer=_Discarding())
