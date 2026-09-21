"""Contract of the ecology's error root and of what a log line may carry.

Tier 2. These checks guard two conventions rather than any behaviour behind
them: that a failure crossing a provider boundary arrives as one catchable
family with the original still attached, and that the context on a log line is
locatable, UTC and incapable of carrying a secret.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import get_args

import pytest
import structlog
from structlog.testing import CapturingLogger

from liq.evolution.ecology import errors, logctx
from liq.evolution.ecology.errors import EcologyError, translating
from liq.evolution.ecology.logctx import (
    LOGGABLE_KEYS,
    REDACTED,
    STAGES,
    ContextBinder,
    LogContext,
    Stage,
    redact_unlisted,
)
from liq.evolution.errors import LiqEvolutionError

BAR_TS = datetime(2026, 9, 19, 13, 30, tzinfo=UTC)

# A provider message shaped like the ones that leak: a credential in a URL.
PROVIDER_MESSAGE = "GET https://user:hunter2@feed.example/bars failed"

# Plan vocabulary that must not appear in anything shipped.
PLAN_VOCABULARY = re.compile(r"\bphase\b|\bV\d+(?:\.\d+)*\b", re.IGNORECASE)

SHIPPED_MODULES = (errors, logctx)


class _WindowUnavailable(EcologyError):
    """A leaf of the kind a slice defines; this module supplies one to test on."""


class _ProviderTimeout(Exception):
    """Stands in for whatever a provider library raises."""


# --- the error root and the translation convention ---------------------------


def test_the_ecology_root_extends_the_platform_root() -> None:
    """Ecology failures are catchable as platform failures, not as a rival family."""
    assert issubclass(EcologyError, LiqEvolutionError)
    assert EcologyError.__bases__ == (LiqEvolutionError,)


def test_no_leaf_types_are_shipped_ahead_of_the_code_that_raises_them() -> None:
    """Only the root and the convention are exported; a catalogue is not."""
    assert set(errors.__all__) == {"EcologyError", "translating"}


def test_a_translated_failure_keeps_the_original_as_its_cause() -> None:
    """The traceback still shows what actually broke."""
    original = _ProviderTimeout(PROVIDER_MESSAGE)
    with (
        pytest.raises(_WindowUnavailable) as caught,
        translating(
            _ProviderTimeout, into=_WindowUnavailable, action="reading a window"
        ),
    ):
        raise original
    assert caught.value.__cause__ is original


def test_a_translated_failure_names_the_provider_type_and_the_attempt() -> None:
    """A reader learns what failed and what was being done, from the message alone."""
    with (
        pytest.raises(_WindowUnavailable) as caught,
        translating(
            _ProviderTimeout, into=_WindowUnavailable, action="reading a window"
        ),
    ):
        raise _ProviderTimeout(PROVIDER_MESSAGE)
    assert str(caught.value) == "_ProviderTimeout while reading a window"


def test_a_translated_failure_does_not_copy_the_provider_message() -> None:
    """A credential quoted by a provider does not travel into the new message."""
    with (
        pytest.raises(_WindowUnavailable) as caught,
        translating(
            _ProviderTimeout, into=_WindowUnavailable, action="reading a window"
        ),
    ):
        raise _ProviderTimeout(PROVIDER_MESSAGE)
    assert "hunter2" not in str(caught.value)
    assert PROVIDER_MESSAGE not in str(caught.value)


def test_an_ecology_failure_passes_through_untranslated() -> None:
    """A failure already described once is not wrapped again at every level."""
    already = _WindowUnavailable("no window at this instant")
    with (
        pytest.raises(_WindowUnavailable) as caught,
        translating(Exception, into=_WindowUnavailable, action="reading a window"),
    ):
        raise already
    assert caught.value is already
    assert caught.value.__cause__ is None


def test_a_failure_the_provider_was_not_said_to_raise_is_left_alone() -> None:
    """A defect in the adapter's own code is not dressed up as a provider failure."""
    with (
        pytest.raises(ZeroDivisionError),
        translating(
            _ProviderTimeout, into=_WindowUnavailable, action="reading a window"
        ),
    ):
        _ = 1 / 0


def test_a_block_that_succeeds_is_untouched() -> None:
    """Translation is a boundary, not a step in the happy path."""
    seen = []
    with translating(
        _ProviderTimeout, into=_WindowUnavailable, action="reading a window"
    ):
        seen.append("ran")
    assert seen == ["ran"]


# --- the log context ---------------------------------------------------------


def test_the_context_keys_are_the_four_that_locate_a_line() -> None:
    """The allow-list is exactly the context, and it does not say `phase`."""
    expected = {"agent_id", "bar_ts", "run_id", "stage"}
    assert set(LOGGABLE_KEYS) == expected
    assert not [key for key in LOGGABLE_KEYS if "phase" in key]


def test_the_stage_vocabulary_is_closed_and_agrees_with_its_type() -> None:
    """The runtime tuple and the static type cannot drift apart unnoticed."""
    assert set(STAGES) == set(get_args(Stage.__value__))
    assert set(STAGES) == {"feature", "size", "fill", "learn", "allocate"}


@pytest.mark.parametrize("stage", STAGES)
def test_every_named_stage_is_accepted(stage: Stage) -> None:
    """Each step of the forward pass can label a line."""
    assert LogContext(run_id="r1", bar_ts=BAR_TS, stage=stage).stage == stage


def test_a_stage_outside_the_vocabulary_is_refused() -> None:
    """A misspelling fails loudly instead of never matching a filter."""
    with pytest.raises(ValueError, match="stage must be one of"):
        LogContext(run_id="r1", bar_ts=BAR_TS, stage="sizing")  # type: ignore[arg-type]


def test_a_naive_timestamp_is_refused() -> None:
    """A timestamp with no zone is not UTC, whatever it was meant to be."""
    with pytest.raises(ValueError, match="timezone-aware"):
        LogContext(run_id="r1", bar_ts=datetime(2026, 9, 19, 13, 30), stage="fill")


def test_a_timestamp_at_another_offset_is_refused() -> None:
    """A local time cannot be emitted at all, not merely discouraged."""
    local = datetime(2026, 9, 19, 9, 30, tzinfo=timezone(timedelta(hours=-4)))
    with pytest.raises(ValueError, match="must be UTC"):
        LogContext(run_id="r1", bar_ts=local, stage="fill")


def test_the_rendered_timestamp_shows_its_offset() -> None:
    """The output states UTC rather than implying it."""
    rendered = LogContext(run_id="r1", bar_ts=BAR_TS, stage="fill").as_event_dict()
    assert rendered["bar_ts"] == "2026-09-19T13:30:00+00:00"


def test_an_unnamed_run_is_refused() -> None:
    """A line that cannot say which run it came from does not locate anything."""
    with pytest.raises(ValueError, match="run_id"):
        LogContext(run_id="", bar_ts=BAR_TS, stage="fill")


def test_an_empty_agent_id_is_refused() -> None:
    """Absent and empty are different claims; only absent is allowed."""
    with pytest.raises(ValueError, match="agent_id"):
        LogContext(run_id="r1", bar_ts=BAR_TS, stage="fill", agent_id="")


def test_a_population_wide_line_omits_the_agent_key() -> None:
    """Computing the shared feature view belongs to no single agent."""
    rendered = LogContext(run_id="r1", bar_ts=BAR_TS, stage="feature").as_event_dict()
    assert "agent_id" not in rendered
    assert set(rendered) == {"bar_ts", "run_id", "stage"}


def test_a_per_agent_line_carries_the_agent_key() -> None:
    """Two agents' lines at one decision point are told apart without reading them."""
    rendered = LogContext(
        run_id="r1", bar_ts=BAR_TS, stage="size", agent_id="a-7"
    ).as_event_dict()
    assert rendered["agent_id"] == "a-7"
    assert set(rendered) <= LOGGABLE_KEYS


def test_the_context_has_nowhere_to_put_a_credential() -> None:
    """The first half of the allow-list: the context holds these fields and no others."""
    with pytest.raises(TypeError):
        LogContext(  # type: ignore[call-arg]
            run_id="r1", bar_ts=BAR_TS, stage="fill", api_token="sk-live-DEADBEEF"
        )


def test_a_context_refuses_mutation() -> None:
    """A bound context is not edited on its way through the pipeline."""
    context = LogContext(run_id="r1", bar_ts=BAR_TS, stage="fill")
    with pytest.raises(AttributeError):
        context.run_id = "r2"  # type: ignore[misc]


# --- the allow-list, end to end through a rendered line -----------------------


def _capturing_logger() -> tuple[CapturingLogger, ContextBinder]:
    """Return a sink and a logger whose chain starts with the redactor."""
    sink = CapturingLogger()
    return sink, structlog.wrap_logger(
        sink,
        processors=[
            redact_unlisted,
            structlog.processors.KeyValueRenderer(sort_keys=True),
        ],
    )


def test_a_credential_shaped_field_never_reaches_the_rendered_line() -> None:
    """The second half of the allow-list: the keyword route is closed too."""
    sink, logger = _capturing_logger()
    context = LogContext(run_id="r1", bar_ts=BAR_TS, stage="fill", agent_id="a-7")
    context.bind(logger).info(
        "fill_accounted",
        api_token="sk-live-DEADBEEF",
        payload={"raw": "quote blob"},
    )
    line = sink.calls[0].args[0]
    assert "sk-live-DEADBEEF" not in line
    assert "quote blob" not in line
    assert f"api_token='{REDACTED}'" in line
    assert f"payload='{REDACTED}'" in line


def test_the_context_itself_survives_the_redactor() -> None:
    """Fail-closed must not mean fail-blank: the four keys are what a line is for."""
    sink, logger = _capturing_logger()
    context = LogContext(run_id="r1", bar_ts=BAR_TS, stage="fill", agent_id="a-7")
    context.bind(logger).info("fill_accounted", secret="x")
    line = sink.calls[0].args[0]
    assert "run_id='r1'" in line
    assert "agent_id='a-7'" in line
    assert "stage='fill'" in line
    assert "bar_ts='2026-09-19T13:30:00+00:00'" in line
    assert "event='fill_accounted'" in line


def test_the_redactor_fails_closed_on_a_field_nobody_anticipated() -> None:
    """A deny-list would have to have guessed this name; an allow-list does not."""
    redacted = redact_unlisted(None, "info", {"frobnicator_handle": "hunter2"})
    assert redacted == {"frobnicator_handle": REDACTED}


def test_the_redactor_keeps_the_key_so_the_refusal_is_visible() -> None:
    """A line shows that something was offered and refused, rather than hiding it."""
    redacted = redact_unlisted(None, "info", {"run_id": "r1", "headers": {"a": "b"}})
    assert redacted == {"run_id": "r1", "headers": REDACTED}


def test_a_bound_logger_satisfies_the_binder_contract() -> None:
    """Binding is structural, so the ecology does not name a logging library."""
    _, logger = _capturing_logger()
    assert isinstance(logger, ContextBinder)


# --- shipped vocabulary ------------------------------------------------------


@pytest.mark.parametrize("module", SHIPPED_MODULES, ids=lambda m: m.__name__)
def test_no_shipped_source_carries_plan_vocabulary(module: object) -> None:
    """Plan numbering lives in the plan, never in code, comments or docstrings."""
    source = Path(module.__file__).read_text(encoding="utf-8")  # type: ignore[attr-defined]
    assert PLAN_VOCABULARY.search(source) is None
