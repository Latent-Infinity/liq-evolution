"""Evidence that the scaling the wish is formed from reaches no later bar.

One claim, and it stopped being a claim about a diagnostic. An estimator that
standardises what it is shown is running a preprocessing step on the inputs, and
a preprocessing step fitted on statistics that include the bar being processed
is look-ahead as surely as reading the bar itself would be — the normalised
feature at bar *t* then depends on bar *t*. This is the FR-10 rule applied to a
step the walking skeleton did not have.

**Why the wish is asserted on and not only the number.** The standardisation
used to be an internal detail of an update nothing read, so a shift in its
window moved a statistic and nothing else. It is now what the decision rule is
formed from: the reading a wish is compared against its entry gene is the
learned weights against *these* values. So the second check below reconstructs
the wish from each of the two orderings and requires the shipped one to be the
causal reconstruction and to differ from the peeking one. A normalisation
look-ahead now moves exposure, and the evidence has to be able to say so.

**How it is shown, and why not by inspection.** The check does not read the
implementation. It takes what the update actually consumed at each decision
point and compares it with two independent reimplementations of the same
arithmetic that differ only in ordering: one accumulating the statistics after
standardising a bar, one before. Matching the first and differing from the
second is the whole claim. The second is also, exactly, the statistics window
shifted one bar forward, so the "shift it and watch the signal move" half and
the "make the harness peek and watch it go red" half are the same mechanism run
once.

**Nothing here is market data.** The decision points are the declared
arithmetic ramp; see `tests/support/level_shift_stream.py` for what that costs
and what the approved tape would have added. The claim is about an ordering
inside the estimator, which is a property of the code rather than of any
instrument, and is why it can be asserted here at all.
"""

from __future__ import annotations

from collections.abc import Sequence

from liq.evolution.ecology import agent, config
from tests.support import level_shift_stream as stream
from tests.support import state_norm_probe as probe

#: The forgetting factor the agent under test carries. The standardisation is
#: required to forget at the same rate as the weights, so the reference below
#: is built with this same number rather than with one of its own.
FORGETTING = 0.99

#: The entry genes the exposure half of the claim is swept over. A declared
#: grid rather than a chosen value, spanning the range the pass's own readings
#: cover, so that "the shift can move an exposure" is established over
#: something stated in advance rather than over a threshold picked because it
#: produced the answer.
ENTRY_GENES_SWEPT = tuple(round(-1.75 + 0.25 * step, 2) for step in range(15))


def _walked() -> tuple[stream.Trajectory, tuple[float, ...]]:
    """One agent's consumed inputs over the declared stream, and the raw feature."""
    state = agent.PopulationState.founded(
        (stream.birth(stream.QUICK_TO_FORGET, forgetting=FORGETTING),),
        learning=config.LearningConfig(),
    )
    windows = stream.windows()
    walked = stream.walk(state, windows)
    raw = tuple(
        window.features[stream.INSTRUMENT][stream.FEATURE] for window in windows
    )
    return walked[stream.QUICK_TO_FORGET], raw


def test_normalisation_shift_changes_signal() -> None:
    """Shifting the statistics window one bar forward moves the standardised signal.

    Three assertions, in the order they matter. What the update consumed is
    what prior-bar statistics produce. It is *not* what statistics reaching the
    current bar produce — so the window is strictly behind, and the check can
    tell the two apart rather than being blind to the difference. And the
    difference is of the order the shift implies: one bar of a
    forgetting-weighted window, largest where the window is shortest.
    """
    trajectory, raw = _walked()
    causal = probe.reference_standardised(raw, forgetting=FORGETTING, peek=False)
    peeking = probe.reference_standardised(raw, forgetting=FORGETTING, peek=True)

    to_causal = _largest_gap(trajectory.standardised, causal)
    to_peeking = _largest_gap(trajectory.standardised, peeking)
    between = _largest_gap(causal, peeking)

    assert between > 0.0, (
        "the reference cannot tell a causal window from one shifted a bar "
        "forward, so matching it would prove nothing"
    )
    assert to_causal < between / 1e6, (
        f"what the update consumed sits {to_causal} from the prior-bar "
        f"standardisation and the two standardisations are only {between} "
        "apart, so it is not the causal one to within the arithmetic's own "
        "noise"
    )
    assert to_peeking > between / 2.0, (
        f"what the update consumed sits {to_peeking} from a standardisation "
        "allowed to see the bar it is scaling, which is not far enough to say "
        "the statistics stop short of it"
    )

    early = max(abs(a - b) for a, b in zip(causal[:20], peeking[:20], strict=True))
    late = max(abs(a - b) for a, b in zip(causal[-20:], peeking[-20:], strict=True))
    assert early > late, (
        f"one bar moved the standardised signal by {early} early in the pass "
        f"and {late} late in it; a forgetting-weighted window is shortest at "
        "the start, so the shift has to matter most there"
    )


def _largest_gap(left: Sequence[float], right: Sequence[float]) -> float:
    """The biggest absolute difference between two trajectories, bar for bar.

    Compared by size rather than for equality because one side is computed
    over a whole population at once and the other one value at a time, and
    those two orders of the same arithmetic differ in the last bit. The claim
    is about which standardisation was used, and a last-bit difference is not
    a different standardisation — so the check asks whether the gap is of the
    order of rounding or of the order of the thing being excluded.
    """
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def test_the_wish_is_formed_from_the_causal_standardisation() -> None:
    """The exposure the agent wished for is the one the prior-bar scaling implies.

    The same two reference orderings as above, carried through to the decision.
    Each bar's wish is reconstructed from the weight the agent actually carried
    when it formed that wish and the reading each ordering implies, and
    compared with the wish it actually formed. The shipped sequence has to *be*
    the causal reconstruction, bar for bar — which is the whole of what makes
    the ordering a property of the exposures rather than of a diagnostic.
    """
    trajectory, raw = _walked()
    causal = probe.reference_standardised(raw, forgetting=FORGETTING, peek=False)

    assert trajectory.wanted == _wishes(
        trajectory.in_force, causal, gene=stream.ENTRY_AT
    ), (
        "the exposure the agent wished for is not the exposure the prior-bar "
        "standardisation implies, so either the wish reads a different scaling "
        "or this reconstruction is of a different rule"
    )


def test_the_shift_can_move_an_exposure_and_not_only_a_statistic() -> None:
    """A one-bar shift changes what is *held*, at entry genes the rule allows.

    The companion the check above needs, and it is deliberately quantified over
    a declared grid of entry genes rather than over the one the stream carries,
    because of a limit that is measured here rather than assumed away.

    **The shift is sign-preserving, and at an entry gene of exactly zero it is
    therefore invisible to the wish.** Folding the current bar into a
    forgetting-weighted window before scaling by it multiplies the centred
    value by ``m·λ / (m·λ + 1)``, which is strictly between zero and one — so
    the peeking reading is the causal reading shrunk towards nothing, never
    reflected through it. A wish that is a sign test cannot see a shrinkage.
    Measured: at the stream's own entry gene of zero, **0 of 800** wishes move.
    That is exactly the weakness the 2026-09-23 Oracle recorded against this
    binding — a differential shift standing behind a sentence with an absolute
    clause — now measured at the decision rather than at a statistic, and it is
    why this binding is owed an absolute oracle before anything signal-bearing
    runs on it.

    What the grid establishes is the other half: at entry genes inside the
    range the reading actually covers, the shift *does* move exposure, so the
    reconstruction is not blind to it and the check above is asserted over
    something that could fail. Both counts are reported.
    """
    trajectory, raw = _walked()
    causal = probe.reference_standardised(raw, forgetting=FORGETTING, peek=False)
    peeking = probe.reference_standardised(raw, forgetting=FORGETTING, peek=True)

    at_the_streams_own_gene = _parting(
        trajectory.in_force, causal, peeking, gene=stream.ENTRY_AT
    )
    assert at_the_streams_own_gene == 0, (
        f"{at_the_streams_own_gene} wishes moved at an entry gene of "
        f"{stream.ENTRY_AT}, where the arithmetic says none can: this "
        "docstring's account of why the shift is sign-preserving is wrong and "
        "the limit recorded against this binding has to be restated"
    )

    moved = {
        gene: _parting(trajectory.in_force, causal, peeking, gene=gene)
        for gene in ENTRY_GENES_SWEPT
    }
    parting = {gene: count for gene, count in moved.items() if count}
    assert parting, (
        f"across {len(ENTRY_GENES_SWEPT)} declared entry genes spanning the "
        "readings this pass produced, shifting the statistics window one bar "
        "forward moved no exposure anywhere; the ordering this fact pins could "
        "then be wrong without a single trade changing"
    )
    assert sum(moved.values()) < len(ENTRY_GENES_SWEPT) * len(causal), (
        "every wish at every declared gene moved, which is not a shift of one "
        "bar in a forgetting-weighted window but a different signal"
    )


def _parting(
    weights: Sequence[float],
    causal: Sequence[float],
    peeking: Sequence[float],
    *,
    gene: float,
) -> int:
    """How many decision points the two orderings wish differently at."""
    here = _wishes(weights, causal, gene=gene)
    there = _wishes(weights, peeking, gene=gene)
    return sum(1 for left, right in zip(here, there, strict=True) if left != right)


def _wishes(
    weights: Sequence[float], standardised: Sequence[float], *, gene: float
) -> tuple[float, ...]:
    """The exposure each reading implies, under one entry gene.

    The rule, restated over one feature that is switched on at every decision
    point: the learned weight against the standardised reading, compared
    strictly above the entry gene. Restated rather than imported, because a
    reconstruction that called the implementation would agree with it whatever
    either of them did.
    """
    return tuple(
        agent.FULL_EXPOSURE if weight * value > gene else 0.0
        for weight, value in zip(weights, standardised, strict=True)
    )


def test_the_first_bar_is_standardised_against_nothing() -> None:
    """At the first decision point there is no prior bar, and none is invented.

    The boundary case is the one where peeking is most tempting and hardest to
    see: a standardiser with no history either reports the bar against itself,
    which is a division by a statistic the bar created, or reports nothing to
    scale by. The second is the only causal answer.
    """
    trajectory, _ = _walked()
    assert trajectory.standardised[0] == 0.0
