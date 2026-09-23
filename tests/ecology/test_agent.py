"""How the rule behaves on genomes and views this repository can construct.

The evidence that a wish is a reproducible function of a genome and a broadcast
lives in the consumer repository, beside the tape it is asserted over. What is
checked here is the rule's own contract on readings the declared placeholder
ramp can produce: a genome that cannot be read, a feature switched off, a
feature the view does not carry, and the two sides of the entry gene.
"""

from __future__ import annotations

import pytest

from liq.evolution.ecology.adapters import NullBarSource
from liq.evolution.ecology.agent import (
    ENTRY_THRESHOLD,
    FULL_EXPOSURE,
    MASK_PREFIX,
    SWITCHED_ON_AT,
    WEIGHT_PREFIX,
    Agent,
)
from liq.evolution.ecology.types import BarWindow, Genome

AGENT_ID = "ramp-agent"
GENE_SCHEMA_VERSION = "ramp-genes-1"

#: The feature the declared ramp broadcasts a level under.
LEVEL = "level"

#: A bar no reading of the ramp reaches.
BEYOND_ANY_READING = 1e12


def _window() -> BarWindow:
    """One decision point from the declared placeholder ramp."""
    return next(iter(NullBarSource().windows()))


def _instrument(window: BarWindow) -> str:
    """The single instrument the ramp carries a bar for."""
    return next(iter(window.bars))


def _agent(**genes: float) -> Agent:
    """An agent carrying exactly the genes named."""
    return Agent(
        agent_id=AGENT_ID,
        genome=Genome(genes=dict(genes), schema_version=GENE_SCHEMA_VERSION),
    )


def test_a_reading_above_the_entry_gene_wants_the_whole_account() -> None:
    """Clearing the gene is wanted in full; there is nothing between that and flat."""
    window = _window()
    agent = _agent(
        **{
            f"{MASK_PREFIX}{LEVEL}": 1.0,
            f"{WEIGHT_PREFIX}{LEVEL}": 1.0,
            ENTRY_THRESHOLD: 0.0,
        }
    )

    wish = agent.intend(window, _instrument(window))

    assert wish.target_exposure == FULL_EXPOSURE
    assert wish.agent_id == AGENT_ID
    assert wish.as_of == window.as_of


def test_a_reading_that_does_not_clear_the_entry_gene_wants_nothing() -> None:
    """The gene is a bar to be cleared, not a target to be approached."""
    window = _window()
    agent = _agent(
        **{
            f"{MASK_PREFIX}{LEVEL}": 1.0,
            f"{WEIGHT_PREFIX}{LEVEL}": 1.0,
            ENTRY_THRESHOLD: BEYOND_ANY_READING,
        }
    )

    assert agent.intend(window, _instrument(window)).target_exposure == 0.0


def test_a_feature_switched_off_is_not_read_and_needs_no_weight() -> None:
    """A mask below the cut takes its feature out of the reading entirely."""
    window = _window()
    agent = _agent(
        **{
            f"{MASK_PREFIX}{LEVEL}": SWITCHED_ON_AT / 2.0,
            ENTRY_THRESHOLD: 0.0,
        }
    )

    assert agent.intend(window, _instrument(window)).target_exposure == 0.0


def test_an_instrument_the_decision_point_carries_no_view_for_reads_as_nothing() -> (
    None
):
    """An absent view is a reading of nothing, never a reach for one elsewhere."""
    window = _window()
    agent = _agent(
        **{
            f"{MASK_PREFIX}{LEVEL}": 1.0,
            f"{WEIGHT_PREFIX}{LEVEL}": 1.0,
            ENTRY_THRESHOLD: 0.0,
        }
    )

    wish = agent.intend(window, "AN_INSTRUMENT_THIS_POINT_DOES_NOT_CARRY")

    assert wish.target_exposure == 0.0
    assert wish.instrument == "AN_INSTRUMENT_THIS_POINT_DOES_NOT_CARRY"


def test_a_genome_with_no_entry_gene_cannot_form_a_wish() -> None:
    """Where the bar sits is inherited; there is no reading that defaults it."""
    window = _window()
    agent = _agent(**{f"{MASK_PREFIX}{LEVEL}": 1.0, f"{WEIGHT_PREFIX}{LEVEL}": 1.0})

    with pytest.raises(KeyError, match=ENTRY_THRESHOLD):
        agent.intend(window, _instrument(window))


def test_a_feature_switched_on_without_a_weight_cannot_form_a_wish() -> None:
    """How much a feature counts is inherited too, and is never defaulted."""
    window = _window()
    agent = _agent(**{f"{MASK_PREFIX}{LEVEL}": 1.0, ENTRY_THRESHOLD: 0.0})

    with pytest.raises(KeyError, match=f"{WEIGHT_PREFIX}{LEVEL}"):
        agent.intend(window, _instrument(window))
