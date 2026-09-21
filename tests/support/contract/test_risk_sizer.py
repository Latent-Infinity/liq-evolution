"""The contract every mandate is run against.

Tier 2. One outcome is the whole of what is checked here: what the agent may
hold, and every bound that bound it, together. Bounds are not all of one kind
and the contract holds a mandate to the difference. A bound on direction — a
wish the mandate permits in no size at all — is refused whole, and no target
comes back with it. A bound on magnitude is applied and recorded: the reduced
exposure is the one that would be traded and scored, so it is carried, and the
bound that reduced it is named beside it.

What is forbidden throughout is the third thing: a number quietly smaller than
the wish with nothing recorded. That scores an agent on a position it did not
choose and leaves no trace of the mandate having acted, which is why every
check below that permits a smaller target also demands the record that explains
it.

Which bounds an adapter holds is its own configuration, so the three probes an
adapter is checked against are declared by the class that registers it rather
than fixed here. They default to the shape of this programme's mandate —
long-only, with a cap on gross exposure — and a mandate of a different shape
names its own.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import UTC, datetime
from types import MappingProxyType
from typing import ClassVar

import pytest

from liq.evolution.ecology import (
    AccountState,
    Intent,
    PositionTarget,
    Rejection,
    RiskSizer,
    SizingOutcome,
)
from liq.evolution.ecology.adapters import NullRiskSizer

# A stable code is compared and counted, so it carries no whitespace and no
# prose. The shape, not any particular vocabulary, is what is contracted.
REASON_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")

AGENT = "contract-agent"
INSTRUMENT = "CONTRACT-A"
OTHER_INSTRUMENT = "CONTRACT-B"
AS_OF = datetime(2000, 1, 1, tzinfo=UTC)

# Declared probes, not market data: wishes spanning nothing, a modest holding,
# the whole account and plainly more than the account, in both directions.
WANTED_EXPOSURES = (0.0, 0.25, -0.25, 1.0, -1.0, 4.0, -4.0)

FLAT_ACCOUNT = AccountState(
    agent_id=AGENT,
    as_of=AS_OF,
    equity=1.0,
    exposures=MappingProxyType({}),
    costs_charged=0.0,
)
ENGAGED_ACCOUNT = AccountState(
    agent_id=AGENT,
    as_of=AS_OF,
    equity=1.0,
    exposures=MappingProxyType({OTHER_INSTRUMENT: 0.9}),
    costs_charged=0.0,
)
ACCOUNTS = (FLAT_ACCOUNT, ENGAGED_ACCOUNT)
ACCOUNT_IDS = ("flat", "engaged")


def wish(exposure: float) -> Intent:
    """Return the probe intent that wants ``exposure`` in the probe instrument."""
    return Intent(
        agent_id=AGENT,
        instrument=INSTRUMENT,
        target_exposure=exposure,
        as_of=AS_OF,
    )


class RiskSizerContract:
    """What a mandate promises, whichever bounds it happens to hold.

    An adapter joins the contract by subclassing this and naming a factory that
    builds it. No suite is copied to add one::

        class TestMyRiskSizer(RiskSizerContract):
            adapter_factory = staticmethod(MyRiskSizer)

    A mandate whose bounds do not bind where the declared probes below expect
    them to names its own alongside the factory::

        class TestWideRiskSizer(RiskSizerContract):
            adapter_factory = staticmethod(WideRiskSizer)
            oversized_wish = 40.0
    """

    adapter_factory: ClassVar[Callable[[], RiskSizer]]

    #: A wish in a direction this mandate permits, small enough that no bound
    #: it holds binds on a flat account.
    permitted_wish: ClassVar[float] = 0.25
    #: A wish in a direction this mandate permits, too large for it to allow
    #: whole on a flat account.
    oversized_wish: ClassVar[float] = 4.0
    #: A wish in a direction this mandate permits in no size at all.
    forbidden_wish: ClassVar[float] = -0.25

    @pytest.fixture
    def adapter(self) -> RiskSizer:
        """The adapter under test, built fresh for this test."""
        return type(self).adapter_factory()

    def test_the_adapter_satisfies_the_port(self, adapter: RiskSizer) -> None:
        """Conformance is of the object handed over, not of a declared class."""
        assert isinstance(adapter, RiskSizer)

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("wanted", WANTED_EXPOSURES)
    def test_every_decision_is_one_well_formed_outcome(
        self, adapter: RiskSizer, wanted: float, account: AccountState
    ) -> None:
        """There is no other outcome: no silence, no error, no bare number."""
        outcome = adapter.size(wish(wanted), account)
        assert isinstance(outcome, SizingOutcome)
        assert outcome.target is None or isinstance(outcome.target, PositionTarget)
        assert isinstance(outcome.rejections, tuple)
        assert all(isinstance(item, Rejection) for item in outcome.rejections)

    def test_an_unbound_wish_is_permitted_whole(self, adapter: RiskSizer) -> None:
        """Where nothing binds, what comes back is the agent's own decision."""
        intent = wish(type(self).permitted_wish)
        outcome = adapter.size(intent, FLAT_ACCOUNT)
        assert outcome.rejections == ()
        assert outcome.target is not None
        assert outcome.target.target_exposure == intent.target_exposure

    def test_an_oversized_wish_is_constrained_and_the_bound_recorded(
        self, adapter: RiskSizer
    ) -> None:
        """A bound on size is the mandate working: it reduces, and says so.

        The reduced exposure is what would be traded and therefore what the
        agent is scored on, so it has to come back rather than be discarded —
        and it has to come back with the bound that produced it, or the score
        would read as the agent's own choice.
        """
        intent = wish(type(self).oversized_wish)
        outcome = adapter.size(intent, FLAT_ACCOUNT)
        assert outcome.target is not None
        assert abs(outcome.target.target_exposure) < abs(intent.target_exposure)
        assert outcome.target.target_exposure * intent.target_exposure >= 0.0
        assert outcome.rejections

    def test_a_forbidden_wish_is_refused_whole(self, adapter: RiskSizer) -> None:
        """A bound on direction is a breach of the mandate, not an overshoot."""
        outcome = adapter.size(wish(type(self).forbidden_wish), FLAT_ACCOUNT)
        assert outcome.target is None
        assert outcome.rejections

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("wanted", WANTED_EXPOSURES)
    def test_a_target_unlike_the_wish_names_what_changed_it(
        self, adapter: RiskSizer, wanted: float, account: AccountState
    ) -> None:
        """No exposure is adjusted silently: an altered number carries its reason."""
        intent = wish(wanted)
        outcome = adapter.size(intent, account)
        if outcome.target is not None and (
            outcome.target.target_exposure != intent.target_exposure
        ):
            assert outcome.rejections

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("wanted", WANTED_EXPOSURES)
    def test_permitting_nothing_still_names_a_reason(
        self, adapter: RiskSizer, wanted: float, account: AccountState
    ) -> None:
        """An agent is never stood down without the record of why."""
        outcome = adapter.size(wish(wanted), account)
        if outcome.target is None:
            assert outcome.rejections

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("wanted", WANTED_EXPOSURES)
    def test_a_permitted_target_never_exceeds_or_reverses_the_wish(
        self, adapter: RiskSizer, wanted: float, account: AccountState
    ) -> None:
        """A mandate may take exposure away; it may not add or invert any."""
        intent = wish(wanted)
        outcome = adapter.size(intent, account)
        if outcome.target is not None:
            assert abs(outcome.target.target_exposure) <= abs(intent.target_exposure)
            assert outcome.target.target_exposure * intent.target_exposure >= 0.0

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("wanted", WANTED_EXPOSURES)
    def test_every_recorded_bound_names_a_stable_code(
        self, adapter: RiskSizer, wanted: float, account: AccountState
    ) -> None:
        """Refusals are counted and compared across runs rather than read."""
        outcome = adapter.size(wish(wanted), account)
        for rejection in outcome.rejections:
            assert REASON_CODE.fullmatch(rejection.reason)
            assert isinstance(rejection.detail, str)

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("wanted", WANTED_EXPOSURES)
    def test_a_decision_is_attributed_to_the_intent_that_asked_for_it(
        self, adapter: RiskSizer, wanted: float, account: AccountState
    ) -> None:
        """A decision is never re-attributed to another agent, instrument or instant."""
        intent = wish(wanted)
        outcome = adapter.size(intent, account)
        decided: list[PositionTarget | Rejection] = list(outcome.rejections)
        if outcome.target is not None:
            decided.append(outcome.target)
        assert decided
        for part in decided:
            assert part.agent_id == intent.agent_id
            assert part.instrument == intent.instrument
            assert part.as_of == intent.as_of

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("wanted", WANTED_EXPOSURES)
    def test_the_same_question_gets_the_same_answer(
        self, adapter: RiskSizer, wanted: float, account: AccountState
    ) -> None:
        """Applying a mandate is a decision, not a draw: it repeats exactly."""
        first = adapter.size(wish(wanted), account)
        second = adapter.size(wish(wanted), account)
        assert first == second

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    def test_deciding_leaves_the_intent_and_the_account_alone(
        self, adapter: RiskSizer, account: AccountState
    ) -> None:
        """The mandate reads what it is handed; it does not adjust it in place."""
        intent = wish(0.5)
        exposures = dict(account.exposures)
        adapter.size(intent, account)
        assert intent == wish(0.5)
        assert dict(account.exposures) == exposures

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    def test_a_permitted_exposure_is_a_finite_number(
        self, adapter: RiskSizer, account: AccountState
    ) -> None:
        """What an agent may hold is a number, never a sentinel standing for one."""
        for wanted in WANTED_EXPOSURES:
            outcome = adapter.size(wish(wanted), account)
            if outcome.target is not None:
                assert math.isfinite(outcome.target.target_exposure)


class TestNullRiskSizer(RiskSizerContract):
    """The stand-in, held to the same contract as any provider-backed mandate."""

    adapter_factory = staticmethod(NullRiskSizer)
