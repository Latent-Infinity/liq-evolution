"""The one order a decision point is taken in: settle what was earned, then decide.

Everything that walks agents over history does the same three things at every
decision point, and in the same order, whatever else it does around them:

1. **Settle.** What the reading formed at the previous decision point turned
   out to be worth is shown to whoever formed it, stamped at *this* decision
   point — the first instant it could be known at. At the first decision point
   of a pass there is no reading behind it and nothing is shown.
2. Whatever the caller does with the bar itself happens next. An evaluation
   account carries what it held and acts on the target it formed a bar ago; a
   demonstration that only reports what agents wanted does nothing here.
3. **Decide.** Only then is the next reading formed, from weights that have
   already been told what the last one earned.

A loop that decided before it settled would form a wish from weights that had
not yet been told what the last one earned, which is a different rule; a loop
that settled against anything but the previous close would be learning from a
different quantity. Written once, here, so that the evaluation account and
anything that measures the rule without an account cannot be taking decision
points in two orders, or settling two different quantities.

What this does not do is price the decision point, walk the history or keep any
account. Which instrument is followed, and whether a decision point that does
not price it is a defect or merely a bar to pass over, is the caller's to say;
walking forward once is :func:`liq.evolution.ecology.driver.walk`'s.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from liq.evolution.ecology.types import Bar, BarWindow, UtcTimestamp

__all__ = ["DecisionStep", "what_the_reading_earned"]


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


def _nothing() -> None:
    """Do nothing with the bar between settling and deciding."""


@dataclass
class DecisionStep:
    """Where the last reading was formed, and the order the next one is taken in.

    Mutable and private to whoever walks with it: one per pass, advanced one
    decision point at a time.

    Attributes:
        previous_close: The close the last reading was formed at, or ``None``
            before the first decision point of the pass.
        previous_as_of: The decision instant the last reading was formed at, or
            ``None`` before the first decision point of the pass. What an
            update settled at this decision point consumed.
    """

    previous_close: float | None = None
    previous_as_of: UtcTimestamp | None = None

    def take[DecidedT](
        self,
        window: BarWindow,
        bar: Bar,
        *,
        settle: Callable[[UtcTimestamp, float], object],
        decide: Callable[[], DecidedT],
        meanwhile: Callable[[], object] = _nothing,
    ) -> DecidedT:
        """Take one decision point: settle, account for the bar, then decide.

        Args:
            window: The decision point.
            bar: The bar ``window`` prices the followed instrument at. Handed
                in rather than looked up, because what an unpriced decision
                point means is the caller's to decide.
            settle: Shows what the last reading earned, stamped at
                ``window.as_of``. Not called at the first decision point.
            decide: Forms the next reading. Called after ``settle`` and
                ``meanwhile``, so it reads weights already told what the last
                reading earned.
            meanwhile: What the caller does with the bar between the two. While
                it runs, :attr:`previous_close` and :attr:`previous_as_of` still
                describe the previous decision point.

        Returns:
            Whatever ``decide`` returned.
        """
        if self.previous_close is not None:
            settle(
                window.as_of, what_the_reading_earned(self.previous_close, bar.close)
            )
        meanwhile()
        decided = decide()
        self.previous_close = bar.close
        self.previous_as_of = window.as_of
        return decided
