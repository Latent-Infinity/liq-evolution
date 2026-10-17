"""The contract every portfolio-risk adapter is run against.

Tier 2. The aggregate book is constrained as one batch: every living agent's
share of it, netted into one target, against the book's own account at one
decision point. What is contracted is the shape every adapter shares, whichever
bounds it holds. The batch is a true batch, so the order the agents are handed
in changes nothing. A target that differs from the netted one names every bound
that changed it, so a reduction or a refusal is never silent. The adapter says
what it applies and what it deliberately does not, each with its reason. And a
book that holds more than one instrument is refused, loudly, because every bound
recorded as not binding was decided on a book of one.

Which bounds bind is the adapter's configuration, so these probes are declared
rather than chosen to make any bound fire: books spanning nothing, part of the
book, the whole of it and more, a short one, and accounts at and below their
high-water mark, latched and not. The null adapter applies nothing and says so.

These bases are also loaded by the consumer under a package name of their own,
where ``tests`` is the consumer's tree, so a base imports nothing from any
``tests`` package. Every boundary name is reached through the package by name,
so the module collects even where one is missing.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, ClassVar

import pytest

from liq.evolution import ecology
from liq.evolution.ecology import AccountState, adapters, ports

# A stable code is compared and counted, so it carries no whitespace and no
# prose. The shape, not any particular vocabulary, is what is contracted.
REASON_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")

INSTRUMENT = "CONTRACT-A"
OTHER_INSTRUMENT = "CONTRACT-B"
AS_OF = datetime(2000, 1, 1, tzinfo=UTC)

# Declared probes, not market data. Each living agent's share of the book, as a
# fraction of book equity, keyed by agent: nothing, part of the book, the whole
# of it, more than it, and a share set that nets short.
SHARE_SETS = (
    {"agent-a": 0.0, "agent-b": 0.0},
    {"agent-a": 0.25, "agent-b": 0.125, "agent-c": 0.0},
    {"agent-a": 0.5, "agent-b": 0.5},
    {"agent-a": 1.0, "agent-b": 0.5},
    {"agent-a": 0.25, "agent-b": -0.5},
)
SHARE_IDS = ("flat", "part", "whole", "over", "net-short")

# What the book holds at the decision point, and what it is worth against the
# high-water mark: at the mark, and a fifth below it.
HELD = (0.0, 0.5)
EQUITIES = (1.0, 0.8)
HIGH_WATER_MARK = 1.0


def boundary(name: str) -> Any:
    """Reach one boundary name through the package, so a missing one is named."""
    return getattr(ecology, name)


def book_account(
    *, held: float, equity: float, other: float | None = None
) -> AccountState:
    """The book's own account at the probe instant."""
    exposures = {INSTRUMENT: held}
    if other is not None:
        exposures[OTHER_INSTRUMENT] = other
    return AccountState(
        agent_id=boundary("BOOK_ACCOUNT_ID"),
        as_of=AS_OF,
        equity=equity,
        exposures=MappingProxyType(exposures),
        costs_charged=0.0,
    )


def netted_book(
    shares: dict[str, float],
    *,
    held: float = 0.0,
    equity: float = 1.0,
    latched: bool = False,
    other: float | None = None,
) -> Any:
    """The probe book: ``shares`` netted, against the book's own account."""
    return boundary("NettedBook")(
        as_of=AS_OF,
        instrument=INSTRUMENT,
        shares=MappingProxyType(dict(shares)),
        netted=math.fsum(shares.values()),
        account=book_account(held=held, equity=equity, other=other),
        high_water_mark=HIGH_WATER_MARK,
        de_risk_latched=latched,
    )


def every_book() -> list[Any]:
    """Every declared probe book."""
    return [
        netted_book(shares, held=held, equity=equity, latched=latched)
        for shares in SHARE_SETS
        for held in HELD
        for equity in EQUITIES
        for latched in (False, True)
    ]


class PortfolioRiskContract:
    """What a portfolio-risk adapter promises, whichever bounds it holds.

    An adapter joins the contract by subclassing this and naming a factory that
    builds it. No suite is copied to add one::

        class TestMyPortfolioRisk(PortfolioRiskContract):
            adapter_factory = staticmethod(MyPortfolioRisk)
    """

    adapter_factory: ClassVar[Callable[[], Any]]

    @pytest.fixture
    def adapter(self) -> Any:
        """The adapter under test, built fresh for this test."""
        boundary("PortfolioRisk")
        return type(self).adapter_factory()

    def test_the_adapter_satisfies_the_port(self, adapter: Any) -> None:
        """Conformance is of the object handed over, not of a declared class."""
        assert isinstance(adapter, boundary("PortfolioRisk"))

    def test_every_decision_is_one_well_formed_outcome(self, adapter: Any) -> None:
        """There is no other outcome: no silence, no bare number."""
        for book in every_book():
            outcome = adapter.constrain(book)
            assert isinstance(outcome, boundary("ConstrainedBook"))
            assert outcome.as_of == book.as_of
            assert outcome.instrument == book.instrument
            assert outcome.netted == book.netted
            assert math.isfinite(outcome.target)

    @pytest.mark.parametrize("shares", SHARE_SETS, ids=SHARE_IDS)
    def test_the_batch_is_a_true_batch(
        self, adapter: Any, shares: dict[str, float]
    ) -> None:
        """Handing the agents in another order changes no book target."""
        forward = netted_book(shares, held=0.5, equity=0.8)
        backward = netted_book(
            dict(reversed(list(shares.items()))), held=0.5, equity=0.8
        )
        assert list(forward.shares) != list(backward.shares) or len(shares) < 2
        assert adapter.constrain(forward) == adapter.constrain(backward)

    def test_a_target_unlike_the_netted_one_names_what_changed_it(
        self, adapter: Any
    ) -> None:
        """No book is reduced or refused silently: an altered number has a reason."""
        for book in every_book():
            outcome = adapter.constrain(book)
            if outcome.target != book.netted:
                assert outcome.rejections

    def test_every_recorded_bound_is_the_books_and_names_a_stable_code(
        self, adapter: Any
    ) -> None:
        """Reductions and refusals are counted and compared, and belong to the book."""
        for book in every_book():
            for rejection in adapter.constrain(book).rejections:
                assert REASON_CODE.fullmatch(rejection.reason)
                assert isinstance(rejection.detail, str)
                assert rejection.agent_id == boundary("BOOK_ACCOUNT_ID")
                assert rejection.instrument == book.instrument
                assert rejection.as_of == book.as_of

    def test_a_de_risk_event_records_the_override(self, adapter: Any) -> None:
        """Where the de-risk path fires, the event says what it replaced and with what."""
        for book in every_book():
            outcome = adapter.constrain(book)
            if outcome.de_risk is not None:
                assert outcome.de_risk.target_before == book.netted
                assert outcome.de_risk.target_after == outcome.target
                assert REASON_CODE.fullmatch(outcome.de_risk.reason)

    def test_the_same_question_gets_the_same_answer(self, adapter: Any) -> None:
        """Constraining a book is a decision, not a draw: it repeats exactly."""
        for book in every_book():
            assert adapter.constrain(book) == adapter.constrain(book)

    def test_what_is_applied_and_what_is_not_are_stated(self, adapter: Any) -> None:
        """Every bound is named as applied, or as not applied with its reason."""
        applied = adapter.bound()
        withheld = adapter.not_bound()
        assert isinstance(applied, tuple)
        assert isinstance(withheld, tuple)
        assert all(REASON_CODE.fullmatch(name) for name in applied)
        for name, reason in withheld:
            assert REASON_CODE.fullmatch(name)
            assert isinstance(reason, str)
            assert reason.strip()
        assert not set(applied) & {name for name, _ in withheld}
        assert applied or withheld, "an adapter that states nothing has said nothing"
        assert adapter.bound() == applied
        assert adapter.not_bound() == withheld

    @pytest.mark.parametrize("other", [0.25, -0.25])
    def test_a_book_of_more_than_one_instrument_is_refused(
        self, adapter: Any, other: float
    ) -> None:
        """Every bound decided on a book of one would stand on a false premise."""
        book = netted_book(SHARE_SETS[1], held=0.25, other=other)
        with pytest.raises(
            getattr(ports, "MoreThanOneInstrument"),  # noqa: B009
            match="one instrument",
        ):
            adapter.constrain(book)

    def test_a_flat_holding_elsewhere_is_still_one_instrument(
        self, adapter: Any
    ) -> None:
        """An instrument named at no exposure is not held, so the book is still one."""
        book = netted_book(SHARE_SETS[1], held=0.25, other=0.0)
        assert isinstance(adapter.constrain(book), boundary("ConstrainedBook"))


def _null_portfolio_risk() -> Any:
    return getattr(adapters, "NullPortfolioRisk")()  # noqa: B009


class TestNullPortfolioRisk(PortfolioRiskContract):
    """The stand-in, held to the same contract as any provider-backed adapter."""

    adapter_factory = staticmethod(_null_portfolio_risk)


def test_the_null_adapter_applies_nothing_and_says_so() -> None:
    """Every book comes back as it was netted, and the adapter states that it binds nothing."""
    adapter = _null_portfolio_risk()
    for book in every_book():
        outcome = adapter.constrain(book)
        assert outcome.target == book.netted
        assert outcome.rejections == ()
        assert outcome.de_risk is None
    assert adapter.bound() == ()
    assert adapter.not_bound()
