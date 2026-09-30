"""One agent's evaluation account, joined through the mandate and the model.

This is the lower level of the two-level accounting model: one agent, its own
account at a normalised notional, scored on what it actually held. The aggregate
ledger — allocation weights, netting, the portfolio-level mandate — is a
different ledger and is not here. Scoring an agent on its own account is what
keeps its fitness from depending on the allocation its fitness is about to
determine.

**Normalised notional.** Every agent opens with the same equity, so two agents'
figures are comparable to each other and neither is a figure about a book. The
account is a fraction of itself: exposure is a signed fraction of equity, a
charge is a fraction of equity, and no currency amount appears anywhere.

**What this module decides is nothing.** A wish is formed by the agent, what may
be held is decided by the mandate behind :class:`~liq.evolution.ecology.ports.RiskSizer`,
what is reached is decided by the model behind
:class:`~liq.evolution.ecology.ports.ExecutionSimulator`, and what it costs is
decided by the cost scenario that model was handed by name. This module joins
them in order and writes down what happened. There is no sizing arithmetic here,
no fill mechanics here and no cost number here — not as a default, not as a
fallback, not as a rounding rule — because each of those is a decision somebody
else has already made and a second copy of it would be a second answer.

**The acting bar is not the forming bar.** A target formed at one decision point
is acted on at the *next* one. A target acted on inside the bar it was formed on
would fill at prices that printed before the decision, so the execution model
refuses it outright rather than absorbing it, and this loop is what keeps the
two a bar apart: the target a decision point produces is held and handed to the
bar that follows it.

**How a bar is accounted, in full.** Three steps, in the order they happen:

1. What was held over the gap is carried to this bar's **open**, which is where
   the execution model acts. A position held from the last close to this open
   earns that move and nothing else.
2. The target formed a bar ago is acted on there, and whatever it is charged is
   charged there, on the equity as it then stands.
3. What is held *after* that is carried to the bar's **close**.

A bar that traded nothing therefore steps close to close, which is the familiar
form. Only a bar that traded differs, and it differs in the direction that stops
a newly established position being credited with a move it was not held over.

**Realised, never wished.** The exposure the account goes on holding is read
back from the account state the execution model hands out, not from the target
that was sent. Where a target is not reached the account holds what it held, is
charged nothing, and is scored on that — which is the whole of the claim this
account exists to make good.

**Why the event trace is emitted here.** Each decision point records what its
learning and fitness updates consumed: the decision point before it, or nothing
at the first, where there is no prior outcome to be shown. The record lives with
the account rather than with the walk because the walk knows nothing about
agents, accounts or scores, and a loop that had to would stop being one loop.
What the trace makes checkable is an ordering that cannot be checked by reading
code: a fitness update told the outcome of the bar it is scoring has been handed
the answer, and so has everything downstream of it. No cull, birth or allocation
decision is taken yet, so nothing *acts* on the refreshed score; the ordering is
established before the decisions that will consume it, because a trace begun
only when its consumer arrived would be asserting about a first run nobody
checked.

**The learning line is now a consumption and not only a record.** The agent's
rule reads what it has learned, so an account that never told it what its
readings earned would run an agent deciding from a cold start for the whole
pass. The settlement happens at the same point the trace is written: the
reading formed at the previous decision point is paired with the move the
instrument made over the bar it was acted on, and shown to the agent stamped at
this instant, which is the first instant it could be known at. The agent
refuses an outcome stamped at the instant its reading was formed, so the
pairing is protected by the thing being updated rather than by this loop
remembering to be careful.
"""

from __future__ import annotations

from typing import Protocol

from liq.evolution.ecology.config import EcologyConfig
from liq.evolution.ecology.driver import walk
from liq.evolution.ecology.errors import EcologyError
from liq.evolution.ecology.logctx import SILENT, Event, RunLog
from liq.evolution.ecology.ports import BarSource, ExecutionSimulator, RiskSizer
from liq.evolution.ecology.types import (
    AgentId,
    BarWindow,
    Fill,
    InstrumentId,
    Intent,
    NotFilled,
    SizingOutcome,
    UtcTimestamp,
)

__all__ = [
    "FITNESS",
    "FLAT",
    "LEARNING",
    "NOTHING_CHARGED",
    "AccountEvent",
    "DecisionPointNotPriced",
    "Evaluation",
    "IntentFormer",
    "evaluate",
    "what_the_reading_earned",
]

#: Kind of event: what one agent was shown of an outcome so that it could update
#: on it.
LEARNING = "learning"

#: Kind of event: what one agent's score was refreshed from.
FITNESS = "fitness"

#: What an account holds in an instrument it is not in.
FLAT = 0.0

#: What an account has been charged before anything has been acted on.
NOTHING_CHARGED = 0.0


def _what_the_mandate_decided(sized: SizingOutcome) -> Event:
    """Which of the three well-formed sizing shapes this outcome is.

    Three, not two. A wish permitted whole and a wish reduced to what the
    mandate allows are both targets and would read alike under one label, and
    the difference between them is the whole of what the mandate did.
    """
    if sized.target is None:
        return "intent_refused"
    return "intent_bound" if sized.rejections else "intent_sized"


def _what_execution_did(outcome: Fill | NotFilled) -> Event:
    """
    Whether acting on a target reached it.

    Read from the outcome's type rather than by comparing exposures, because that is where the distinction is made: an outcome that traded
    nothing is a different value from one that traded, and it cannot be mistaken for a fill of nothing.
    """
    return "target_not_reached" if isinstance(outcome, NotFilled) else "target_reached"


def what_the_reading_earned(previous_close: float, close: float) -> float:
    """
    What the reading formed a bar ago turned out to be worth.

    One definition, in one place, because two would be two answers. A wish formed at one decision point is acted on over the bar that follows
    it, so what that reading was worth is the move the instrument made across that bar — from the close the reading was formed at to the close
    it is being settled at — and it is known at the later of the two instants and no earlier.

    It is the instrument's own move rather than the agent's realised profit, and the difference matters: an agent scored on its own profit while
    flat would be shown zero at every bar it stayed out of, and would learn from its own inaction that nothing predicts anything. What the
    update is being asked is what holding a full position over that bar would have earned, which is a fact about the tape and not about the
    wish.

    *Declared by the executor, not by the requirement.* The PRD says the update learns "from realized outcomes" and does not say which realised
    outcome; the Oracle of 2026-09-27 explicitly took no view on what the update should predict. This is the narrowest quantity that satisfies
    the sentence and keeps the decision rule interpretable — the reading becomes a prediction of the next bar's move and the entry gene becomes
    a hurdle on it — and it is named here so that changing it is one edit with one home.

    Args:
    previous_close: The close the reading was formed at.
    close: The close the bar it was acted on over finished at.

    Returns:
    float: The fractional move between the two.
    """
    return close / previous_close - 1.0


class IntentFormer(Protocol):
    """
    Whatever forms one agent's wish at a decision point, and learns from it.

    Named structurally rather than as the agent type, because what this account needs of whoever forms its wishes is an identity and a rule —
    and a stand-in that exists to want something the mandate refuses is exactly what the long-only claim has to be checked against. A nominal
    dependency here would make that check impossible to write without the ecology's own agent learning to misbehave.

    Two methods rather than one, and the second is not an optional extra. The shipped rule forms its wish from what the agent has learned, so an
    account that asked for wishes and never said what they earned would run an agent that decides from a cold start forever. Being shown an
    outcome is therefore part of forming the next wish, and a stand-in that learns nothing says so by accepting the call and doing nothing with
    it.
    """

    agent_id: AgentId

    def intend(self, window: BarWindow, instrument: InstrumentId) -> Intent:
        """Return what this agent wants to hold in ``instrument`` at ``window``."""
        ...

    def observe(self, as_of: UtcTimestamp, outcome: float) -> object:
        """
        Show this agent what the wish it last formed went on to earn.

        Called once per decision point after the first, stamped at the instant the outcome became known — which is strictly after the decision point
        the reading was formed at, because what holding a reading earned cannot be known before the bar it was acted on over has finished.
        """
        ...


class DecisionPointNotPriced(EcologyError):
    """
    The decision point carried no completed bar for the instrument followed.

    An account follows one instrument, and a decision point that prices it is what makes the account expressible there: without a bar there is
    no open to act at and no close to carry to. Marking the position at the last price it had would forward-fill a mark, and holding the account
    still would report an unchanged equity as though the market had not moved, so neither is done.

    It is raised rather than recorded, for the reason a refused target is: it says the account was pointed at an instrument this stream does not
    carry, which is a wiring defect and not a market outcome.
    """


def evaluate(
    source: BarSource,
    config: EcologyConfig,
    *,
    agent: IntentFormer,
    instrument: InstrumentId,
    sizer: RiskSizer,
    simulator: ExecutionSimulator,
    opening_equity: float,
    log: RunLog = SILENT,
) -> Evaluation:
    """Run one agent over ``source`` once and return its own evaluation account.

    The pass is driven by the walk rather than by a loop written here, so the
    forward-once discipline the walk enforces holds of this account for free: a
    decision point offered out of order, twice, or in a segment already left
    stops the run where it is refused rather than being scored.

    Args:
        source: Where decision points come from. Only the port is used.
        config: How the pass is driven, and the record of what it was allowed to
            be.
        agent: Whose wishes are run. An identity and a rule for forming them.
        instrument: The single instrument the account follows. Every decision
            point of the pass must price it.
        sizer: The mandate. What may be held, and every bound that bound.
        simulator: The execution model. What was reached, and what it charged
            under the scenario it was handed by name.
        opening_equity: What the account opens with, in normalised notional
            units. The same for every agent, so two agents' scores are
            comparable to each other.
        log: Where the two boundaries this account crosses write what they did:
            what the mandate decided about each wish, and whether acting on
            each permitted target reached it. Both lines name the agent, since
            both concern one member rather than the population. The log is
            handed to the walk as well, so one run's lines come from one log
            and not from two configured apart. Silent until a run wires a
            writer, and silent is the same run: nothing an account computes is
            read from, or conditioned on, what it wrote down.

    Returns:
        Evaluation: The account as of every decision point, what it traded, what
        was refused, what each bar's updates consumed, and what it scored.

    Raises:
        ValueError: If the account opens with nothing. A score is what an
            account earned on what it opened with, and an account that opened
            with nothing has no such figure.
        DecisionPointNotPriced: If a decision point carries no completed bar for
            ``instrument``.
    """
    if opening_equity <= 0.0:
        raise ValueError(
            f"an evaluation account must open with something: {opening_equity} "
            "leaves no notional for an exposure to be a fraction of and no "
            "figure for a score to be earned on"
        )
    account = _EvaluationAccount(
        agent=agent,
        instrument=instrument,
        sizer=sizer,
        simulator=simulator,
        equity=opening_equity,
        log=log,
    )
    history = walk(source, config, visit=account.visit, log=log)
    return Evaluation(
        agent_id=agent.agent_id,
        instrument=instrument,
        cost_scenario_id=simulator.cost_scenario_id,
        opening_equity=opening_equity,
        states=tuple(account.states),
        fills=tuple(account.fills),
        rejections=tuple(account.rejections),
        events=tuple(account.events),
        history=history,
    )


from .account_state import AccountEvent, Evaluation, _EvaluationAccount  # noqa: E402
