"""The fixed-share aggregator's own cases, each exactly as the frozen record states it.

The record is ``liq-docs/plans/agent-ecology-aggregator-contract.md`` at its
frozen sha256, read through :mod:`tests.support.weight_oracle`: cold start
(§2.4(5), §2.7(1)), a newborn's entry (§2.7(2)), a death's redistribution
(§2.4(4) step 3, §2.7(3)), missing outcomes and the sleeping-expert rule
(§2.5), ties and order (§2.8), no clipping (§2.6(1)) and normalisation
(§2.6(2)–(4)), the warm-up of the reference scale (§2.2), and the refusals
(§2.5(5), §2.9(2)–(3)). Expected values come from the oracle's independent
rebuild, compared under the record's §7 rule; nothing here is market data.

The module is imported inside each test body, so this file collects while the
module is absent and every case then fails naming it.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from tests.support import weight_oracle as oracle

MODULE = "liq.evolution.ecology.fixed_share"


def _module():
    return importlib.import_module(MODULE)


def _record() -> oracle.Record:
    return oracle.read_record()


def _aggregator():
    fs = _module()
    return fs, fs.FixedShare(
        contract_sha256=oracle.FROZEN_SHA256,
        eta=fs.ETA,
        share_rate=fs.SHARE_RATE,
    )


def _as_mapping(state) -> dict[str, float]:
    return dict(zip(state.agent_ids, state.weights.tolist(), strict=True))


def _warm(fs, aggregator, ids, *, reference=1.0):
    """A state whose scale is defined, built by consuming warm-up decision points."""
    state = aggregator.start(ids)
    nothing = np.zeros(len(ids), dtype=bool)
    for _ in range(fs.SCALE_WARMUP):
        state = aggregator.update(
            state,
            living=ids,
            returns=np.zeros(len(ids)),
            recorded=nothing,
            reference=reference,
        )
    return state


def _assert_on_simplex(fs, state) -> None:
    count = len(state.agent_ids)
    assert np.all(state.weights >= fs.SHARE_RATE / count)
    assert abs(math.fsum(state.weights.tolist()) - 1.0) <= oracle.tolerance(count)


# --- the record's values --------------------------------------------------------


def test_the_module_carries_the_frozen_record_values() -> None:
    fs = _module()
    contract = _record().contract
    assert fs.CONTRACT_SHA256 == oracle.FROZEN_SHA256
    assert contract["contract_id"] == fs.CONTRACT_ID
    assert contract["eta"] == fs.ETA
    assert contract["share_rate"] == fs.SHARE_RATE
    assert contract["scale_warmup"] == fs.SCALE_WARMUP
    assert tuple(contract["allocation_keys"]) == fs.ALLOCATION_KEYS  # ty: ignore[invalid-argument-type]


def test_the_record_worked_cases_are_reproduced() -> None:
    fs, aggregator = _aggregator()
    record = _record()
    for case in oracle.worked_cases(record):
        before_ids = tuple(sorted(case.before))
        state = fs.FixedShareState(
            agent_ids=before_ids,
            weights=np.array([case.before[a] for a in before_ids]),
            count=fs.SCALE_WARMUP if case.losses else 0,
            sum_of_squares=float(fs.SCALE_WARMUP) if case.losses else 0.0,
        )
        returns = np.array([-case.losses.get(a, 0.0) for a in case.living])
        recorded = np.array([a in case.losses for a in case.living])
        after = aggregator.update(
            state,
            living=case.living,
            returns=returns,
            recorded=recorded,
            reference=None,
        )
        assert (
            oracle.mismatches(
                dict(case.printed),
                _as_mapping(after),
                newborns=case.newborns,
                floor=case.floor,
                share_rate=fs.SHARE_RATE,
            )
            == []
        ), f"case {case.number}"


# --- cold start, births, deaths --------------------------------------------------


def test_cold_start_is_uniform_over_the_founders() -> None:
    fs, aggregator = _aggregator()
    state = aggregator.start(("f2", "f0", "f1"))
    assert state.agent_ids == ("f0", "f1", "f2")
    assert state.weights.tolist() == [1 / 3] * 3
    assert (state.count, state.sum_of_squares) == (0, 0.0)
    first = aggregator.update(
        state,
        living=("f0", "f1", "f2"),
        returns=np.zeros(3),
        recorded=np.zeros(3, dtype=bool),
        reference=None,
    )
    assert np.allclose(first.weights, 1 / 3, rtol=0, atol=oracle.tolerance(3))


def test_a_newborn_enters_at_exactly_the_share() -> None:
    fs, aggregator = _aggregator()
    state = _warm(fs, aggregator, ("a", "b"))
    after = aggregator.update(
        state,
        living=("a", "b", "c"),
        returns=np.array([0.5, -0.5, 0.0]),
        recorded=np.array([True, True, True]),
        reference=1.0,
    )
    assert after.agent_ids == ("a", "b", "c")
    assert after.weights[2] == fs.SHARE_RATE / 3
    _assert_on_simplex(fs, after)


def test_a_death_returns_its_mass_pro_rata_before_mixing() -> None:
    fs, aggregator = _aggregator()
    state = fs.FixedShareState(
        agent_ids=("a", "b", "c"),
        weights=np.array([0.5, 0.3, 0.2]),
        count=0,
        sum_of_squares=0.0,
    )
    after = aggregator.update(
        state,
        living=("a", "b"),
        returns=np.zeros(2),
        recorded=np.zeros(2, dtype=bool),
        reference=None,
    )
    s = fs.SHARE_RATE
    expected = [(1 - s) * 0.5 / 0.8 + s / 2, (1 - s) * 0.3 / 0.8 + s / 2]
    assert np.allclose(after.weights, expected, rtol=0, atol=oracle.tolerance(2))
    assert "c" not in after.agent_ids


def test_a_population_of_one_carries_all_the_weight() -> None:
    fs, aggregator = _aggregator()
    state = _warm(fs, aggregator, ("a", "b"))
    after = aggregator.update(
        state,
        living=("b",),
        returns=np.array([3.0]),
        recorded=np.array([True]),
        reference=1.0,
    )
    assert after.weights.tolist() == [1.0]


# --- missing outcomes, warm-up, scale -------------------------------------------


def test_an_asleep_agent_keeps_its_weight_through_the_loss_step() -> None:
    fs, aggregator = _aggregator()
    state = _warm(fs, aggregator, ("a", "b", "c"))
    state = fs.FixedShareState(
        agent_ids=state.agent_ids,
        weights=np.array([0.5, 0.3, 0.2]),
        count=state.count,
        sum_of_squares=state.sum_of_squares,
    )
    after = aggregator.update(
        state,
        living=("a", "b", "c"),
        returns=np.array([1.0, -1.0, 0.0]),
        recorded=np.array([True, True, False]),
        reference=1.0,
    )
    rebuilt = oracle.loss_step_and_mix(
        {"a": 0.5, "b": 0.3, "c": 0.2},
        ("a", "b", "c"),
        {"a": -1.0, "b": 1.0},
        eta=fs.ETA,
        share_rate=fs.SHARE_RATE,
    )
    assert (
        oracle.mismatches(
            rebuilt, _as_mapping(after), newborns=frozenset(), share_rate=fs.SHARE_RATE
        )
        == []
    )
    s = fs.SHARE_RATE
    assert after.weights[2] == pytest.approx((1 - s) * 0.2 + s / 3, abs=1e-15)


def test_every_agent_sleeps_through_the_warm_up() -> None:
    fs, aggregator = _aggregator()
    ids = ("a", "b")
    state = fs.FixedShareState(
        agent_ids=ids,
        weights=np.array([0.6, 0.4]),
        count=fs.SCALE_WARMUP - 1,
        sum_of_squares=1.0,
    )
    after = aggregator.update(
        state,
        living=ids,
        returns=np.array([5.0, -5.0]),
        recorded=np.array([True, True]),
        reference=2.0,
    )
    s = fs.SHARE_RATE
    assert np.allclose(
        after.weights,
        [(1 - s) * 0.6 + s / 2, (1 - s) * 0.4 + s / 2],
        rtol=0,
        atol=oracle.tolerance(2),
    )
    assert (after.count, after.sum_of_squares) == (fs.SCALE_WARMUP, 5.0)


def test_a_zero_reference_scale_leaves_every_agent_asleep() -> None:
    fs, aggregator = _aggregator()
    state = _warm(fs, aggregator, ("a", "b"), reference=0.0)
    assert state.sum_of_squares == 0.0
    after = aggregator.update(
        state,
        living=("a", "b"),
        returns=np.array([9.0, -9.0]),
        recorded=np.array([True, True]),
        reference=0.0,
    )
    assert np.allclose(after.weights, 0.5, rtol=0, atol=oracle.tolerance(2))


def test_the_loss_is_scaled_by_the_strictly_prior_reference() -> None:
    fs, aggregator = _aggregator()
    ids = ("a", "b")
    scale = oracle.Scale()
    state = aggregator.start(ids)
    before = dict.fromkeys(ids, 0.5)
    rng = np.random.default_rng(7)
    for t in range(fs.SCALE_WARMUP + 6):
        returns = rng.standard_normal(2) * 0.01
        reference = float(rng.standard_normal() * 0.02)
        state = aggregator.update(
            state,
            living=ids,
            returns=returns,
            recorded=np.ones(2, dtype=bool),
            reference=reference,
        )
        before, scale = oracle.update(
            before,
            ids,
            dict(zip(ids, returns.tolist(), strict=True)),
            scale,
            reference,
            eta=fs.ETA,
            share_rate=fs.SHARE_RATE,
            warmup=fs.SCALE_WARMUP,
        )
        assert (
            oracle.mismatches(
                before,
                _as_mapping(state),
                newborns=frozenset(),
                share_rate=fs.SHARE_RATE,
            )
            == []
        ), t
        assert state.count == scale.count


def test_an_update_with_no_reference_leaves_the_scale_unchanged() -> None:
    fs, aggregator = _aggregator()
    state = _warm(fs, aggregator, ("a", "b"))
    after = aggregator.update(
        state,
        living=("a", "b"),
        returns=np.zeros(2),
        recorded=np.ones(2, dtype=bool),
        reference=None,
    )
    assert (after.count, after.sum_of_squares) == (state.count, state.sum_of_squares)


# --- order, ties, clipping -------------------------------------------------------


def test_the_living_are_held_in_ascending_agent_id_order() -> None:
    fs, aggregator = _aggregator()
    state = _warm(fs, aggregator, ("b", "a", "c"))
    shuffled = aggregator.update(
        state,
        living=("c", "a", "b"),
        returns=np.array([0.3, 0.1, 0.2]),
        recorded=np.ones(3, dtype=bool),
        reference=1.0,
    )
    ordered = aggregator.update(
        state,
        living=("a", "b", "c"),
        returns=np.array([0.1, 0.2, 0.3]),
        recorded=np.ones(3, dtype=bool),
        reference=1.0,
    )
    assert shuffled.agent_ids == ("a", "b", "c")
    assert shuffled.weights.tolist() == ordered.weights.tolist()


def test_equal_outcomes_give_equal_weights_with_no_tie_break() -> None:
    fs, aggregator = _aggregator()
    state = _warm(fs, aggregator, ("a", "b", "c"))
    after = aggregator.update(
        state,
        living=("a", "b", "c"),
        returns=np.full(3, 0.4),
        recorded=np.ones(3, dtype=bool),
        reference=1.0,
    )
    assert after.weights[0] == after.weights[1] == after.weights[2]


def test_a_huge_loss_is_not_clipped_and_the_floor_holds() -> None:
    fs, aggregator = _aggregator()
    state = _warm(fs, aggregator, ("a", "b"))
    after = aggregator.update(
        state,
        living=("a", "b"),
        returns=np.array([-1.0e6, 1.0e6]),
        recorded=np.ones(2, dtype=bool),
        reference=1.0,
    )
    assert after.weights[0] == fs.SHARE_RATE / 2
    assert after.weights[1] == pytest.approx(
        1.0 - fs.SHARE_RATE / 2, abs=oracle.tolerance(2)
    )


def test_the_update_is_deterministic() -> None:
    fs, aggregator = _aggregator()
    state = _warm(fs, aggregator, ("a", "b", "c"))
    args = {
        "living": ("a", "b", "c"),
        "returns": np.array([0.1, -0.2, 0.05]),
        "recorded": np.ones(3, dtype=bool),
        "reference": 0.3,
    }
    first = aggregator.update(state, **args)
    second = aggregator.update(state, **args)
    assert first.weights.tobytes() == second.weights.tobytes()


# --- refusals --------------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_outcome_is_refused(bad: float) -> None:
    fs, aggregator = _aggregator()
    state = _warm(fs, aggregator, ("a", "b"))
    with pytest.raises(fs.NonFiniteAllocationInput, match="outcome"):
        aggregator.update(
            state,
            living=("a", "b"),
            returns=np.array([bad, 0.0]),
            recorded=np.ones(2, dtype=bool),
            reference=1.0,
        )


def test_an_unrecorded_slot_is_never_read() -> None:
    fs, aggregator = _aggregator()
    state = _warm(fs, aggregator, ("a", "b"))
    after = aggregator.update(
        state,
        living=("a", "b"),
        returns=np.array([math.nan, 0.0]),
        recorded=np.array([False, True]),
        reference=1.0,
    )
    _assert_on_simplex(fs, after)


def test_an_integer_mask_is_read_as_the_boolean_mask_it_spells() -> None:
    fs, aggregator = _aggregator()
    state = fs.FixedShareState(
        agent_ids=("a", "b", "c"),
        weights=np.array([0.5, 0.3, 0.2]),
        count=fs.SCALE_WARMUP,
        sum_of_squares=float(fs.SCALE_WARMUP),
    )
    returns = np.array([0.4, math.nan, -0.1])
    spelled, boolean = (
        aggregator.update(
            state,
            living=("a", "b", "c"),
            returns=returns,
            recorded=mask,
            reference=1.0,
        )
        for mask in (np.array([1, 0, 1]), np.array([True, False, True]))
    )
    assert spelled.agent_ids == boolean.agent_ids
    assert spelled.weights.tobytes() == boolean.weights.tobytes()
    assert (spelled.count, spelled.sum_of_squares) == (
        boolean.count,
        boolean.sum_of_squares,
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_non_finite_reference_is_refused(bad: float) -> None:
    fs, aggregator = _aggregator()
    state = aggregator.start(("a", "b"))
    with pytest.raises(fs.NonFiniteAllocationInput, match="reference"):
        aggregator.update(
            state,
            living=("a", "b"),
            returns=np.zeros(2),
            recorded=np.zeros(2, dtype=bool),
            reference=bad,
        )


def test_a_non_finite_weight_is_refused() -> None:
    fs, aggregator = _aggregator()
    tiny = 1.0e-150
    state = _warm(fs, aggregator, ("a", "b"), reference=tiny)
    with pytest.raises(fs.NonFiniteAllocationInput, match="weight"):
        aggregator.update(
            state,
            living=("a", "b"),
            returns=np.array([1.0e200, 0.0]),
            recorded=np.ones(2, dtype=bool),
            reference=tiny,
        )


def test_errors_are_ecology_errors() -> None:
    fs = _module()
    from liq.evolution.ecology.errors import EcologyError

    for name in ("NonFiniteAllocationInput", "AllocationRefused", "UnusableLiving"):
        assert issubclass(getattr(fs, name), EcologyError)


@pytest.mark.parametrize(
    ("living", "returns", "recorded", "match"),
    [
        ((), np.zeros(0), np.zeros(0, dtype=bool), "no living"),
        (("a", "a"), np.zeros(2), np.zeros(2, dtype=bool), "twice"),
        (("a", "b", "b"), np.zeros(3), np.zeros(3, dtype=bool), "twice"),
        (("a", "b"), np.zeros(3), np.zeros(2, dtype=bool), "aligned"),
        (("a", "b"), np.zeros(2), np.zeros(3, dtype=bool), "aligned"),
        (("x", "y"), np.zeros(2), np.zeros(2, dtype=bool), "survivor"),
    ],
)
def test_an_unusable_living_set_is_refused(living, returns, recorded, match) -> None:
    fs, aggregator = _aggregator()
    state = aggregator.start(("a", "b"))
    with pytest.raises(fs.UnusableLiving, match=match):
        aggregator.update(
            state, living=living, returns=returns, recorded=recorded, reference=None
        )


def test_starting_with_no_founder_or_a_repeated_founder_is_refused() -> None:
    fs, aggregator = _aggregator()
    with pytest.raises(fs.UnusableLiving, match="no living"):
        aggregator.start(())
    with pytest.raises(fs.UnusableLiving, match="twice"):
        aggregator.start(("a", "a"))
    with pytest.raises(fs.UnusableLiving, match="twice"):
        aggregator.start(("a", "b", "b"))


@pytest.mark.parametrize(
    "changed",
    [
        {"eta": 0.2},
        {"share_rate": 0.0},
        {"share_rate": 0.5},
        {"share_rate": 1.0},
        {"eta": True},
        {"eta": "0.19952623149688797"},
        {"contract_sha256": "0" * 64},
    ],
)
def test_a_value_other_than_the_record_is_refused(changed) -> None:
    fs = _module()
    values = {
        "contract_sha256": fs.CONTRACT_SHA256,
        "eta": fs.ETA,
        "share_rate": fs.SHARE_RATE,
    }
    values.update(changed)
    with pytest.raises(fs.AllocationRefused):
        fs.FixedShare(**values)


def test_the_allocation_table_is_strict() -> None:
    fs = _module()
    frozen = {
        "contract_sha256": fs.CONTRACT_SHA256,
        "eta": fs.ETA,
        "share_rate": fs.SHARE_RATE,
    }
    assert fs.from_allocation(None) is None
    built = fs.from_allocation(frozen)
    assert isinstance(built, fs.FixedShare)
    with pytest.raises(fs.AllocationRefused, match="aggregator"):
        fs.from_allocation({**frozen, "aggregator": "fixed_share"})
    with pytest.raises(fs.AllocationRefused, match="missing"):
        fs.from_allocation({"eta": fs.ETA, "share_rate": fs.SHARE_RATE})
