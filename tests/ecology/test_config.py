"""What a run's configuration refuses to be.

The schema's interesting content is what it will not accept. A run with no
identity cannot be reproduced or withdrawn, and a run configured for a
discontinuity nothing implements would record a policy in its provenance that it
did not follow — which is worse than having no setting at all, because the
record would look deliberate.

That the schema exposes no epoch or replay key is asserted where the fact lives,
against the tape, in the consumer repository. It is not restated here: a second
copy of a claim is a second place for it to drift.
"""

from __future__ import annotations

import dataclasses

import pytest

from liq.evolution.ecology.config import EcologyConfig


def test_a_run_carries_an_identity() -> None:
    """A pass that cannot be named is refused where it is configured."""
    with pytest.raises(ValueError, match="must carry an identity"):
        EcologyConfig(run_id="")


def test_history_carries_across_a_split_boundary_unless_told_otherwise() -> None:
    """Continuity is the default, and it is the reading that is implemented."""
    assert EcologyConfig(run_id="named").window_carries_across_segments is True


def test_a_discontinuity_nothing_implements_is_refused_rather_than_ignored() -> None:
    """Asking for a restart at a fold fails; it is never accepted and disregarded."""
    with pytest.raises(ValueError, match="nothing implements that"):
        EcologyConfig(run_id="named", window_carries_across_segments=False)


def test_a_configuration_cannot_be_changed_once_a_run_is_driven_under_it() -> None:
    """One reader cannot re-point a run's configuration under another."""
    config = EcologyConfig(run_id="named")

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.run_id = "renamed"  # type: ignore[misc]
