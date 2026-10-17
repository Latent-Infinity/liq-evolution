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
import math

import pytest

from liq.evolution.ecology.config import (
    DEFAULT_HISTORY_CAPACITY,
    STATE_BOUND_KEY,
    WEIGHT_NORM_BOUND_KEY,
    EcologyConfig,
    LearningConfig,
    PopulationConfig,
)


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


def test_the_declared_losses_and_bounds_are_keys_of_the_run() -> None:
    """The ceiling and both bounds are configuration a run's digest covers.

    An outcome history's ceiling is a declaration that the oldest outcome will
    be dropped, and a state bound is a declaration about what a run would be
    stopped from doing. Neither is recoverable from a finished run if it lives
    as a default at a call site, so both are keys here and this is the check
    that says so.
    """
    declared = dataclasses.asdict(EcologyConfig(run_id="named"))
    assert declared["population"]["history_capacity"] == DEFAULT_HISTORY_CAPACITY
    assert set(declared["learning"]) == {
        "forgetting_minimum",
        "forgetting_maximum",
        "state_bound",
        "weight_norm_bound",
        "step_scale",
        "step_offset",
        "absolute_penalty",
        "squared_penalty",
    }
    assert STATE_BOUND_KEY == "learning.state_bound"
    assert WEIGHT_NORM_BOUND_KEY == "learning.weight_norm_bound"


def test_an_outcome_history_that_retains_nothing_is_refused() -> None:
    """A ceiling of zero is not a small history, it is the absence of one."""
    with pytest.raises(ValueError, match="retains nothing"):
        PopulationConfig(history_capacity=0)


@pytest.mark.parametrize(
    ("changes", "refused"),
    [
        ({"forgetting_minimum": 0.0}, "strictly inside"),
        ({"forgetting_minimum": 0.99, "forgetting_maximum": 0.5}, "strictly inside"),
        ({"forgetting_maximum": 1.5}, "strictly inside"),
        ({"state_bound": 0.0}, "admits no state at all"),
        ({"weight_norm_bound": -1.0}, "admits no state at all"),
        ({"step_scale": 0.0}, "no step to take"),
        ({"step_offset": -1.0}, "no step to take"),
        ({"absolute_penalty": -1.0}, "rewards a larger weight"),
        ({"squared_penalty": -0.5}, "rewards a larger weight"),
    ],
)
def test_a_learning_configuration_that_means_nothing_is_refused(
    changes: dict[str, float], refused: str
) -> None:
    """Each declared quantity is checked where it is declared, not where it is used.

    A forgetting factor outside (0, 1] is not slow or fast learning, it is
    neither; a bound at or below zero admits no state, so nothing could run
    inside it; a step that divides by zero is not a step; and a negative
    penalty rewards exactly what regularisation exists to discourage.
    """
    with pytest.raises(ValueError, match=refused):
        LearningConfig(**changes)


# ---------------------------------------------------------------------------
# The allocation configuration
# ---------------------------------------------------------------------------
#
# The loop allocates only where it is configured to, and only at the frozen
# aggregator contract's values. The ``[allocation]`` part is the record's closed
# table; the rebalance band is a required parameter with no default, its value
# supplied by whoever composes the run. Both are refused, never repaired.


def _frozen_table() -> dict[str, object]:
    """The record's ``[allocation]`` table, read from the frozen record itself."""
    from tests.support import weight_oracle

    record = weight_oracle.read_record()
    return {
        "contract_sha256": record.sha256,
        "eta": record.contract["eta"],
        "share_rate": record.contract["share_rate"],
    }


def _turnover():
    from tests.support import record_trace

    return record_trace.turnover()


#: A test-local band. The run's value is the composition's to supply.
_BAND = 0.0


def test_no_allocation_is_configured_by_default() -> None:
    """A pass that names no allocation produces no weights and keeps no book."""
    assert EcologyConfig(run_id="named").allocation is None


def test_the_frozen_table_and_a_band_configure_an_allocation() -> None:
    """The record's three keys, by value, and the band beside them."""
    from liq.evolution.ecology.config import AllocationConfig

    configured = AllocationConfig.from_tables(_frozen_table(), band=_BAND)
    table = _frozen_table()
    assert configured.contract_sha256 == table["contract_sha256"]
    assert configured.eta == table["eta"]
    assert configured.share_rate == table["share_rate"]
    assert configured.band == _BAND
    config = EcologyConfig(run_id="named", turnover=_turnover(), allocation=configured)
    assert config.allocation is configured
    assert dataclasses.asdict(config)["allocation"]["band"] == _BAND


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"aggregator": "fixed-share"}, id="unknown-key"),
        pytest.param({"band": _BAND}, id="band-inside-the-record-table"),
        pytest.param({"eta": None}, id="missing-eta"),
        pytest.param({"contract_sha256": None}, id="missing-sha256"),
    ],
)
def test_an_allocation_table_with_an_unknown_or_missing_key_is_refused(
    change: dict[str, object],
) -> None:
    """The table is closed: an extra key is refused, and so is an absent one."""
    from liq.evolution.ecology.config import AllocationConfig
    from liq.evolution.ecology.fixed_share import AllocationRefused

    table = _frozen_table()
    for key, value in change.items():
        if value is None:
            del table[key]
        else:
            table[key] = value
    with pytest.raises(AllocationRefused):
        AllocationConfig.from_tables(table, band=_BAND)


def test_an_allocation_without_its_band_is_refused() -> None:
    """The band has no default: an allocation that does not state one is not built."""
    from liq.evolution.ecology.config import AllocationConfig

    with pytest.raises(TypeError, match="band"):
        AllocationConfig(**_frozen_table())  # ty: ignore[missing-argument]
    with pytest.raises(TypeError, match="band"):
        AllocationConfig.from_tables(_frozen_table())  # ty: ignore[missing-argument]


@pytest.mark.parametrize("band", [-0.1, 1.0, math.nan, math.inf, True])
def test_a_band_outside_its_range_is_refused(band: object) -> None:
    """A band is a fraction of book equity in [0, 1), and a number, not a flag."""
    from liq.evolution.ecology.config import AllocationConfig
    from liq.evolution.ecology.fixed_share import AllocationRefused

    with pytest.raises(AllocationRefused, match="band"):
        AllocationConfig.from_tables(_frozen_table(), band=band)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"eta": 0.1}, id="eta"),
        pytest.param({"share_rate": 0.5}, id="share-rate"),
        pytest.param({"contract_sha256": "0" * 64}, id="sha256"),
    ],
)
def test_an_allocation_value_other_than_the_records_is_refused(
    change: dict[str, object],
) -> None:
    """Only the frozen record's values configure an aggregator."""
    from liq.evolution.ecology.config import AllocationConfig
    from liq.evolution.ecology.fixed_share import AllocationRefused

    with pytest.raises(AllocationRefused):
        AllocationConfig.from_tables({**_frozen_table(), **change}, band=_BAND)
    with pytest.raises(AllocationRefused):
        AllocationConfig(**{**_frozen_table(), **change}, band=_BAND)  # ty: ignore[invalid-argument-type]


def test_an_allocation_without_turnover_is_refused() -> None:
    """Only the steady-state loop allocates; a plain evaluation would ignore it."""
    from liq.evolution.ecology.config import AllocationConfig

    configured = AllocationConfig.from_tables(_frozen_table(), band=_BAND)
    with pytest.raises(ValueError, match="steady-state"):
        EcologyConfig(run_id="named", allocation=configured)


def test_an_allocation_that_is_not_an_allocation_configuration_is_refused() -> None:
    """A raw table handed where the parsed configuration belongs is not read."""
    with pytest.raises(TypeError, match="AllocationConfig"):
        EcologyConfig(
            run_id="named",
            turnover=_turnover(),
            allocation={**_frozen_table(), "band": _BAND},  # ty: ignore[invalid-argument-type]
        )
