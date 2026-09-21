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
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from datetime import UTC, timedelta
from typing import Literal, Protocol, Self, runtime_checkable

from liq.evolution.ecology.types import AgentId, UtcTimestamp

__all__ = [
    "LOGGABLE_KEYS",
    "REDACTED",
    "STAGES",
    "ContextBinder",
    "LogContext",
    "Stage",
    "redact_unlisted",
]

# A step of the per-bar forward pass. Closed: the five steps are the pipeline.
type Stage = Literal["feature", "size", "fill", "learn", "allocate"]

# The same vocabulary at runtime, so a value can be checked and not only typed.
STAGES: tuple[Stage, ...] = ("feature", "size", "fill", "learn", "allocate")

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
