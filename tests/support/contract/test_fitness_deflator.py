"""The contract every fitness deflator is run against.

Tier 2. A population compared on raw fitness rewards whoever was luckiest among
many, so the figure a cull orders by is deflated for how many agents were
compared and for how much independent evidence each one has. Which formula does
that is the adapter's business — and the platform's single statistics authority,
never a formula written in this library. What is contracted is the shape every
deflator shares: one result per sample in the order handed in, the undeflated
figure carried beside the deflated one, the same answer for the same samples, a
window too short or not finite refused rather than scored, and one direction a
deflation can never go — up when more agents are compared against the same
sample — and one thing more evidence can never do: weaken a verdict the agent
already has.

The first is stated weakly, as "does not rise", so a deflator that deflates
nothing satisfies it; the second binds only agents already told apart, which a
deflator that tells nobody apart never has. A real deflator is held to the
strict versions by the repository that registers it.

Each result also carries a standing: whether the agent's figure is told apart
from the population's at the deflator's declared confidence, and in which
direction — worse, indistinguishable, or better. Direction is the deflator's to
say, because a reader that inferred it by comparing figures would be computing a
statistic of its own. Comparing more agents never makes an indistinguishable
agent distinguishable, in either direction.

The samples are windows of the declared level-shift stream's outcomes — the
arithmetic this library's learning evidence already runs on, never an invented
return sequence. Its outcome rule is restated below rather than imported: these
bases are also loaded by the consumer under a package name of their own, where
``tests`` is the consumer's tree, so a base imports nothing from any ``tests``
package. The source of truth is ``tests/support/level_shift_stream.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import combinations
from typing import ClassVar

import pytest

from liq.evolution.ecology import (
    DeflatedFitness,
    FitnessDeflator,
    FitnessSample,
    Standing,
)
from liq.evolution.ecology.adapters import NullFitnessDeflator
from liq.evolution.ecology.errors import EcologyError

# The level-shift stream's outcome rule, restated (see the module docstring):
# a feature cycling through a short odd period, centred on zero, and an outcome
# that is a fixed multiple of it — the multiple changing sign once, at the break.
STREAM_CYCLE = 7
STREAM_BREAK_AT = 400
STREAM_SENSITIVITY_BEFORE = 1.0
STREAM_SENSITIVITY_AFTER = -2.0

# How long each agent's window is, and where the windows start: a length that
# is not a whole number of cycles, so no window's mean is zero by construction,
# and starts either side of the break, so the agents compared do differ.
WINDOW = 10
WINDOW_STARTS = (3, 120, 395, 401, 610)

# One fewer outcome than the shortest window anything can be deflated over.
TOO_SHORT = 1

# The three verdicts a deflator may give an agent against the population it was
# compared with. Only the first licenses a cull; a deflator that cannot tell an
# agent apart in either direction says the second.
WORSE: Standing = "worse"
INDISTINGUISHABLE: Standing = "indistinguishable"
BETTER: Standing = "better"
STANDINGS = (WORSE, INDISTINGUISHABLE, BETTER)


def stream_outcome(index: int) -> float:
    """The level-shift stream's outcome over the bar after decision point ``index``."""
    feature = float(index % STREAM_CYCLE) - float(STREAM_CYCLE // 2)
    sensitivity = (
        STREAM_SENSITIVITY_BEFORE
        if index < STREAM_BREAK_AT
        else STREAM_SENSITIVITY_AFTER
    )
    return sensitivity * feature


def window(start: int, length: int = WINDOW) -> tuple[float, ...]:
    """The stream's outcomes over ``length`` bars from decision point ``start``."""
    return tuple(stream_outcome(index) for index in range(start, start + length))


def sample(
    agent_id: str,
    outcomes: Sequence[float],
    *,
    independent_outcomes: int | None = None,
) -> FitnessSample:
    """One agent's sample, every outcome counted independent unless stated."""
    return FitnessSample(
        agent_id=agent_id,
        outcomes=tuple(outcomes),
        independent_outcomes=(
            len(outcomes) if independent_outcomes is None else independent_outcomes
        ),
    )


def population() -> tuple[FitnessSample, ...]:
    """One sample per declared window start, each for its own agent."""
    return tuple(
        sample(f"deflated-agent-{position}", window(start))
        for position, start in enumerate(WINDOW_STARTS)
    )


def assessed_pairs(
    adapter: FitnessDeflator, own: FitnessSample, *rivals: FitnessSample
) -> list[tuple[DeflatedFitness, DeflatedFitness]]:
    """``own``'s results at every pair of comparison sizes it is assessed at.

    Sizes run from ``own`` alone to ``own`` against every rival, and each pair
    is ordered smaller size first. A size at which ``own`` is unassessed is left
    out, so no pair is made of an absence.
    """
    assessed = [
        result
        for compared in range(len(rivals) + 1)
        if (result := adapter.deflate((own, *rivals[:compared]))[0]).assessed
    ]
    return list(combinations(assessed, 2))


class FitnessDeflatorContract:
    """What a deflator promises, whichever formula it applies.

    An adapter joins the contract by subclassing this and naming a factory that
    builds it. No suite is copied to add one::

        class TestMyFitnessDeflator(FitnessDeflatorContract):
            adapter_factory = staticmethod(MyFitnessDeflator)
    """

    adapter_factory: ClassVar[Callable[[], FitnessDeflator]]

    @pytest.fixture
    def adapter(self) -> FitnessDeflator:
        """The adapter under test, built fresh for this test."""
        return type(self).adapter_factory()

    def test_the_adapter_satisfies_the_port(self, adapter: FitnessDeflator) -> None:
        """Conformance is of the object handed over, not of a declared class."""
        assert isinstance(adapter, FitnessDeflator)

    def test_one_result_per_sample_in_the_order_handed_in(
        self, adapter: FitnessDeflator
    ) -> None:
        """Each result names its agent and how many were compared alongside it."""
        samples = population()
        results = adapter.deflate(samples)
        assert isinstance(results, tuple)
        assert [result.agent_id for result in results] == [
            each.agent_id for each in samples
        ]
        for each, result in zip(samples, results, strict=True):
            assert isinstance(result, DeflatedFitness)
            assert result.compared == len(samples)
            assert result.independent_outcomes == each.independent_outcomes
            assert result.standing in STANDINGS

    def test_nothing_handed_in_is_nothing_deflated(
        self, adapter: FitnessDeflator
    ) -> None:
        """An empty comparison has no members to report on."""
        assert adapter.deflate(()) == ()

    def test_the_undeflated_figure_accompanies_the_deflated_one(
        self, adapter: FitnessDeflator
    ) -> None:
        """Each figure is finite or absent, and an absence is never a verdict.

        A result the deflator cannot assess carries no deflated figure and is
        ``"indistinguishable"``; one with no undeflated figure has nothing to
        deflate either. Full windows with dispersion, as the declared stream's
        are, are assessed by every deflator, so the deflation can be read off.
        """
        results = adapter.deflate(population())
        for result in results:
            for figure in (result.undeflated, result.deflated):
                assert figure is None or math.isfinite(figure)
            if result.deflated is None:
                assert result.standing == INDISTINGUISHABLE
            if result.undeflated is None:
                assert result.deflated is None
        assert all(result.assessed for result in results)

    def test_the_shortest_assessable_window_is_declared(
        self, adapter: FitnessDeflator
    ) -> None:
        """A whole number of outcomes, never below the port's own floor of two."""
        minimum_window = adapter.minimum_window
        assert isinstance(minimum_window, int)
        assert not isinstance(minimum_window, bool)
        assert minimum_window >= 2

    def test_deflation_is_reproducible(self, adapter: FitnessDeflator) -> None:
        """The same samples give the same results."""
        samples = population()
        assert adapter.deflate(samples) == adapter.deflate(samples)

    def test_a_window_too_short_to_deflate_is_refused(
        self, adapter: FitnessDeflator
    ) -> None:
        """Fewer than two outcomes carry no spread, so nothing can be deflated."""
        short = sample("short-agent", window(WINDOW_STARTS[0], TOO_SHORT))
        with pytest.raises(EcologyError):
            adapter.deflate((*population(), short))

    def test_a_non_finite_outcome_is_refused(self, adapter: FitnessDeflator) -> None:
        """A missing number is never scored, imputed or skipped over in silence."""
        outcomes = list(window(WINDOW_STARTS[0]))
        outcomes[WINDOW // 2] = math.nan
        with pytest.raises(EcologyError):
            adapter.deflate((*population(), sample("gap-agent", outcomes)))

    @pytest.mark.parametrize("independent_outcomes", [1, WINDOW + 1])
    def test_an_independent_count_the_window_cannot_hold_is_refused(
        self, adapter: FitnessDeflator, independent_outcomes: int
    ) -> None:
        """Independent outcomes number at least two and at most the window."""
        claimed = sample(
            "overclaimed-agent",
            window(WINDOW_STARTS[0]),
            independent_outcomes=independent_outcomes,
        )
        with pytest.raises(EcologyError):
            adapter.deflate((*population(), claimed))

    def test_more_agents_compared_never_raise_one_agents_figure(
        self, adapter: FitnessDeflator
    ) -> None:
        """The same sample, compared against more rivals, is deflated no less.

        Held across every pair of sizes at which the agent is assessed, not only
        neighbouring ones: an unassessed size in the middle would otherwise let
        a figure that rose across it through.
        """
        for fewer, more in assessed_pairs(adapter, *population()):
            assert more.deflated is not None
            assert fewer.deflated is not None
            assert more.deflated <= fewer.deflated

    def test_more_independent_evidence_never_weakens_a_verdict(
        self, adapter: FitnessDeflator
    ) -> None:
        """Evidence that already tells an agent apart only tells it apart more firmly.

        The window and the comparison are held fixed, so the per-outcome figure
        is unchanged by construction and only the evidence behind it grows. For
        any two independent-outcome counts at which an agent is assessed: a
        "better" standing at the smaller count stays "better" at the larger,
        with a deflated figure no lower; a "worse" standing stays "worse", with
        a deflated figure no higher. Nothing is asserted of an
        "indistinguishable" agent's figure: more evidence makes a deflation
        surer of whichever side of its bar the agent sits on, including the
        side below it, so that figure may fall
        (plans/oracle-v3-worse-standing-2026-10-01.md §3).
        """
        population_now = population()
        for position, own in enumerate(population_now):
            results = []
            for independent_outcomes in range(2, len(own.outcomes) + 1):
                samples = list(population_now)
                samples[position] = sample(
                    own.agent_id,
                    own.outcomes,
                    independent_outcomes=independent_outcomes,
                )
                results.append(adapter.deflate(tuple(samples))[position])
            assert len({result.undeflated for result in results}) == 1
            assessed = [result for result in results if result.assessed]
            for less, more in combinations(assessed, 2):
                assert less.deflated is not None
                assert more.deflated is not None
                if less.standing == BETTER:
                    assert more.standing == BETTER
                    assert more.deflated >= less.deflated
                if less.standing == WORSE:
                    assert more.standing == WORSE
                    assert more.deflated <= less.deflated

    def test_more_agents_compared_never_tell_an_indistinguishable_agent_apart(
        self, adapter: FitnessDeflator
    ) -> None:
        """An agent nobody could tell apart stays so as more rivals are compared.

        Each further comparison is a further chance to be lucky, so it can only
        make a verdict harder to reach — never move an agent from
        indistinguishable toward worse or better with its own sample unchanged.
        Held across every pair of sizes at which the agent is assessed: an agent
        unassessed at one size and assessed at another has gained a verdict
        where it had none, not had one made easier.
        """
        for fewer, more in assessed_pairs(adapter, *population()):
            if fewer.standing == INDISTINGUISHABLE:
                assert more.standing == INDISTINGUISHABLE


class TestNullFitnessDeflator(FitnessDeflatorContract):
    """The stand-in, held to the same contract as the platform-backed deflator."""

    adapter_factory = staticmethod(NullFitnessDeflator)

    def test_the_stand_in_deflates_nothing_and_claims_nothing(
        self, adapter: FitnessDeflator
    ) -> None:
        """It reports the raw figure as deflated, and tells nobody apart.

        A deflator that deflates nothing has no evidence to tell any agent
        apart with, so it says every agent is indistinguishable — and a loop
        wired to it can retire by age but never cull on a figure it never
        deflated.
        """
        for result in adapter.deflate(population()):
            assert result.deflated == result.undeflated
            assert result.standing == INDISTINGUISHABLE

    def test_the_stand_in_declares_the_ports_own_floor(
        self, adapter: FitnessDeflator
    ) -> None:
        """A mean needs no more than the port already refuses below."""
        assert adapter.minimum_window == 2
