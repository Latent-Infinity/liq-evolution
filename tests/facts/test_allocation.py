"""Allocation facts: never top-k, on the simplex, and tracking a switch of best expert.

Three facts, each against the frozen aggregator contract record
(``liq-docs/plans/agent-ecology-aggregator-contract.md``), read only at the
sha256 its freeze names and only through :mod:`tests.support.weight_oracle`:

* ``ECOLOGY.ALLOCATION.NEVER_TOP_K.v1`` (EV-E19) — over every configuration
  path the ``[allocation]`` table can take (the record's ``allocation_keys``
  are the only configuration paths; its other ``[contract]`` labels are fixed
  record fields, freeze record §4(3)), a path either yields no weights, is
  refused, or yields weights that at every decision point sit at or above the
  floor ``s / P`` over exactly the living, with ``s > 0``, and that move when a
  non-leading agent's outcome history moves.
* ``ECOLOGY.ALLOCATION.WEIGHTS_ON_SIMPLEX.v1`` (EV-E82) — finite, non-negative,
  summing to one within ``16 * P * 2**-52`` over exactly the living, nothing
  left on the dead, with births and deaths walked.
* ``ECOLOGY.ALLOCATION.TRACKS_REGIME_SWITCH.v1`` (EV-E20) — on the materialised
  evidence family E, the switched-to expert carries the strictly largest weight
  within the horizon ``H`` the share rate implies, and keeps it to the end of
  its regime.

**Outcome histories for EV-E19 and EV-E82** come from the declared arithmetic
streams S1, S3 and S4 of :mod:`tests.support.turnover_stream` (itself built on
:mod:`tests.support.level_shift_stream`), walked by the real steady-state loop
so the living set changes by births and deaths. Each agent's outcome at a
decision point is the change in its own evaluation account's equity over that
bar, at the normalised notional it opened with; a founder at the first point
and a newborn at its birth point have none. The reference return is the
instrument's close-to-close return. Nothing in them is market data.

**EV-E20's streams are generated (authorized by Task V5.2)** —
``liq-docs/plans/oracle-v5-generated-streams-2026-10-15.md``. They are read
only through the generator :mod:`tests.support.two_regime_stream` and the
manifest under ``tests/fixtures/generated/ecology/two_regime/``, never altered,
non-signal-bearing and never citable. Each stream's seed and generator version
is printed at its point of use. ``H`` is evaluated from the manifest's declared
parameters at the record's frozen ``eta`` and ``share_rate`` before any stream
is run; a regime whose ``H`` exceeds its length is reported and the test stops
there, for a spawned Oracle — ``eta``, ``share_rate`` and E are never changed
to answer it.

The aggregator module is imported inside each test body, so the file collects
while it is absent.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from liq.evolution import ecology
from tests.support import turnover_stream as streams
from tests.support import two_regime_stream as generated
from tests.support import weight_oracle as oracle

MODULE = "liq.evolution.ecology.fixed_share"

#: §2.9(2)'s named refusals; every other ``[contract]`` label is added to them.
NAMED_REFUSALS = (
    "aggregator",
    "method",
    "kind",
    "type",
    "name",
    "top_k",
    "k",
    "alpha",
    "floor",
    "clip",
    "shrinkage",
    "window",
)

#: The evidence family E, as materialised under the manifest pinned below.
FAMILY = Path(__file__).resolve().parents[1] / "fixtures/generated/ecology/two_regime"
MANIFEST_SHA256 = "765f660192f0caad9aa85e5a6cfb41164def0cd56beb6cc803604a6219efc7ed"

#: The horizon each stream of E cites, as its declaration implies. Source:
#: The plan's execution line, recomputed in
#: ``liq-docs/plans/oracle-v5-sufficiency-gaps-2026-10-15.md`` IV6. E05, E06
#: and E08 cite none.
DECLARED_HORIZONS = {"E01": 46, "E02": 23, "E03": 46, "E04": 23, "E07": 46}

#: How far a perturbed outcome is moved, in the outcome's own units.
NUDGE = 1.0e-3


def _module():
    return importlib.import_module(MODULE)


# --- outcome histories over the declared streams --------------------------------


@dataclass(frozen=True)
class _History:
    """A walked population: who lived after each point's turnover, and what was recorded."""

    name: str
    founders: tuple[str, ...]
    living: tuple[tuple[str, ...], ...]
    returns: tuple[Mapping[str, float], ...]
    reference: tuple[float | None, ...]


def _turnover():
    return ecology.steady_state.TurnoverConfig(  # ty: ignore[unresolved-attribute]
        band_lower=14,
        band_upper=20,
        replacement_count=2,
        trigger_period=3,
        minimum_age=6,
        replacement_window=6,
        maximum_replaced_fraction=0.3,
        fitness_window=8,
        parents_per_birth=2,
        inheritance="baldwinian",
        admission_redraws=8,
        seed=11,
    )


def _walk(name: str, windows, founding) -> _History:
    run = streams.steady(
        windows=windows,
        founding=founding,
        turnover=_turnover(),
        deflator=streams.ScriptedDeflator(streams.lowest_is_worse),
    )
    index_of = {window.as_of: i for i, window in enumerate(windows)}
    returns: list[dict[str, float]] = [{} for _ in windows]
    for agent_id, account in run.run.evaluation.accounts.items():
        states = account.states
        for before, after in zip(states, states[1:], strict=False):
            returns[index_of[after.as_of]][agent_id] = (
                after.equity - before.equity
            ) / account.opening_equity
    closes = [w.bars[streams.INSTRUMENT].close for w in windows]
    reference = (None, *(b / a - 1.0 for a, b in zip(closes, closes[1:], strict=False)))
    living = run.watched.living
    return _History(
        name=name,
        founders=tuple(sorted(living[0])),
        living=tuple(tuple(sorted(each)) for each in living[1:]),
        returns=tuple(returns),
        reference=reference,
    )


def _histories() -> tuple[_History, ...]:
    count = 120
    return (
        _walk("S1", streams.priced(count=count), streams.founders(14)),
        _walk("S3", streams.priced(sign=-1, count=count), streams.founders(14)),
        _walk("S4", streams.ramp(count=count), streams.ramp_founders(14)),
    )


def _arrays(history: _History, index: int, nudge: Mapping[str, float] | None = None):
    living = history.living[index]
    recorded = history.returns[index]
    shift = nudge or {}
    returns = np.array([recorded.get(a, 0.0) + shift.get(a, 0.0) for a in living])
    mask = np.array([a in recorded for a in living], dtype=bool)
    return living, returns, mask


def _states(aggregator, history: _History, nudge_at: int = -1, nudge=None) -> Iterator:
    """Every (index, before, after) the aggregator produces walking ``history``."""
    state = aggregator.start(history.founders)
    for index in range(len(history.living)):
        living, returns, mask = _arrays(
            history, index, nudge if index == nudge_at else None
        )
        after = aggregator.update(
            state,
            living=living,
            returns=returns,
            recorded=mask,
            reference=history.reference[index],
        )
        yield index, state, after
        state = after


def _simplex_breaches(history: _History, index, after) -> list[str]:
    """Every way one decision point's vector leaves the simplex over exactly the living."""
    where = f"{history.name}@{index}"
    found = []
    living = history.living[index]
    count = len(living)
    if tuple(after.agent_ids) != living:
        found.append(f"{where}: weights over {after.agent_ids}, living {living}")
        return found
    weights = after.weights
    if weights.dtype != np.float64 or not np.all(np.isfinite(weights)):
        found.append(f"{where}: a weight is not a finite float64")
    if not np.all(weights >= 0.0):
        found.append(f"{where}: a negative weight")
    total = math.fsum(weights.tolist())
    if abs(total - 1.0) > oracle.tolerance(count):
        found.append(f"{where}: weights sum to {total!r}")
    return found


def _rebuild_breaches(fs, history: _History, index, before, after) -> list[str]:
    """Every way one decision point's vector departs from the oracle's rebuild.

    A vector over other agents than the living is reported by
    :func:`_simplex_breaches`, so nothing is rebuilt for it here.
    """
    where = f"{history.name}@{index}"
    living = history.living[index]
    if tuple(after.agent_ids) != living:
        return []
    weights = after.weights
    rebuilt, _ = oracle.update(
        dict(zip(before.agent_ids, before.weights.tolist(), strict=True)),
        living,
        {a: r for a, r in history.returns[index].items() if a in set(living)},
        oracle.Scale(before.count, before.sum_of_squares),
        history.reference[index],
        eta=fs.ETA,
        share_rate=fs.SHARE_RATE,
        warmup=fs.SCALE_WARMUP,
    )
    newborns = frozenset(set(living) - set(before.agent_ids))
    return [
        f"{where}: {each}"
        for each in oracle.mismatches(
            rebuilt,
            dict(zip(after.agent_ids, weights.tolist(), strict=True)),
            newborns=newborns,
            share_rate=fs.SHARE_RATE,
        )
    ]


def _turned_over_after_warm_up(fs, history: _History) -> None:
    """The walk is a precondition: births and deaths after the scale is defined."""
    late = range(fs.SCALE_WARMUP + 1, len(history.living))
    births = sum(len(set(history.living[i]) - set(history.living[i - 1])) for i in late)
    deaths = sum(len(set(history.living[i - 1]) - set(history.living[i])) for i in late)
    assert births and deaths, f"{history.name}: no turnover after the warm-up"


# --- EV-E19 ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Path:
    """One configuration path: a description and the ``[allocation]`` table, or ``None``."""

    describe: str
    table: Mapping[str, object] | None


def _configuration_paths(record: oracle.Record) -> list[_Path]:
    """Every path the record's configuration space admits a loader to be handed."""
    contract = record.contract
    keys = tuple(contract["allocation_keys"])  # ty: ignore[invalid-argument-type]
    frozen: dict[str, object] = {
        "contract_sha256": record.sha256,
        "eta": contract["eta"],
        "share_rate": contract["share_rate"],
    }
    assert set(frozen) == set(keys), "the record's configuration keys moved"
    eta = float(contract["eta"])  # ty: ignore[invalid-argument-type]
    share = float(contract["share_rate"])  # ty: ignore[invalid-argument-type]
    paths = [_Path("no [allocation] table", None), _Path("the frozen table", frozen)]
    paths += [
        _Path(f"without {key}", {k: v for k, v in frozen.items() if k != key})
        for key in keys
    ]
    changed: dict[str, tuple[object, ...]] = {
        "eta": (0.0, -eta, math.nextafter(eta, 1.0), math.inf, math.nan),
        "share_rate": (0.0, 1.0, -share, math.nextafter(share, 0.0), math.nan),
        "contract_sha256": ("0" * 64, record.sha256.upper()),
    }
    paths += [
        _Path(f"{key} = {value!r}", {**frozen, key: value})
        for key, values in changed.items()
        for value in values
    ]
    extras = sorted(set(NAMED_REFUSALS) | (set(contract) - set(keys)))
    paths += [
        _Path(f"with {name}", {**frozen, name: contract.get(name, 0)})
        for name in extras
    ]
    return paths


def _first_free_point(fs, aggregator, history: _History) -> int:
    """The first decision point with a defined loss, no birth, and every agent awake."""
    for index, before, _ in _states(aggregator, history):
        living = history.living[index]
        if (
            before.count >= fs.SCALE_WARMUP
            and set(living) <= set(before.agent_ids)
            and set(living) <= set(history.returns[index])
            and len(living) >= 3
        ):
            return index
    raise AssertionError(f"{history.name}: no decision point with every agent awake")


def _perturbation_breaches(fs, aggregator, history: _History) -> list[str]:
    """Moving a non-leading agent's outcome moves every weight; a warm-up outcome moves none.

    The second half is the record's §2.5(6): an outcome recorded before the
    first decision point at which the loss is defined is consumed by no update,
    so the perturbation property applies only at or after that point.
    """
    found = []
    at = _first_free_point(fs, aggregator, history)
    plain = {i: after for i, _, after in _states(aggregator, history)}
    before = plain[at - 1]
    lowest = before.agent_ids[int(np.argmin(before.weights))]
    nudged = {
        i: after for i, _, after in _states(aggregator, history, at, {lowest: NUDGE})
    }
    unmoved = [
        a
        for a, x, y in zip(
            nudged[at].agent_ids, plain[at].weights, nudged[at].weights, strict=True
        )
        if x == y
    ]
    if unmoved:
        found.append(
            f"{history.name}@{at}: moving {lowest}'s outcome left {unmoved} unmoved"
        )
    early = 5
    assert early < fs.SCALE_WARMUP
    some = next(iter(history.returns[early]), None)
    if some is not None:
        warm = list(_states(aggregator, history, early, {some: NUDGE}))
        moved = [
            i
            for i, _, after in warm
            if after.weights.tobytes() != plain[i].weights.tobytes()
        ]
        if moved:
            found.append(
                f"{history.name}: a warm-up outcome moved the weights at {moved[:3]}"
            )
    return found


def test_no_config_yields_top_k_weights() -> None:
    record = oracle.read_record()
    paths = _configuration_paths(record)
    histories = _histories()
    share = float(record.contract["share_rate"])  # ty: ignore[invalid-argument-type]
    assert share > 0.0, "the share floor's alpha is the share rate and must exceed 0"
    fs = _module()
    accepted = []
    found: list[str] = []
    for path in paths:
        try:
            aggregator = fs.from_allocation(path.table)
        except fs.AllocationRefused:
            continue
        if aggregator is None:
            if path.table is not None:
                found.append(f"{path.describe}: a table that produced no aggregator")
            continue
        accepted.append(path.describe)
        if aggregator.share_rate <= 0.0:
            found.append(f"{path.describe}: share rate {aggregator.share_rate}")
        for history in histories:
            _turned_over_after_warm_up(fs, history)
            for index, before, after in _states(aggregator, history):
                count = len(history.living[index])
                floor = aggregator.share_rate / count
                if not np.all(after.weights >= floor):
                    found.append(
                        f"{path.describe} {history.name}@{index}: below the floor"
                    )
                found += [
                    f"{path.describe} {each}"
                    for each in _simplex_breaches(history, index, after)
                    + _rebuild_breaches(fs, history, index, before, after)
                ]
            found += [
                f"{path.describe} {each}"
                for each in _perturbation_breaches(fs, aggregator, history)
            ]
    assert accepted == ["the frozen table"], f"accepted paths: {accepted}"
    assert not found, "\n".join(found[:20])


def test_a_record_of_other_bytes_turns_the_check_red(tmp_path) -> None:
    altered = tmp_path / "agent-ecology-aggregator-contract.md"
    altered.write_bytes(oracle.RECORD.read_bytes() + b"\n")
    try:
        oracle.read_record(altered)
    except AssertionError as refused:
        assert "not the frozen" in str(refused)
    else:
        raise AssertionError("a record of other bytes was read")


def test_weight_oracle_reproduces_the_record_worked_cases() -> None:
    record = oracle.read_record()
    print(f"comparison rule: {oracle.COMPARISON_RULE}")
    assert oracle.self_check(record) == []


# --- EV-E82 ----------------------------------------------------------------------


def test_weights_are_non_negative_and_sum_to_one_over_the_expert_set() -> None:
    record = oracle.read_record()
    histories = _histories()
    fs = _module()
    aggregator = fs.FixedShare(
        contract_sha256=record.sha256,
        eta=fs.ETA,
        share_rate=fs.SHARE_RATE,
    )
    found: list[str] = []
    for history in histories:
        _turned_over_after_warm_up(fs, history)
        reached = 0
        try:
            for index, before, after in _states(aggregator, history):
                found += _simplex_breaches(history, index, after)
                dead = set(before.agent_ids) - set(history.living[index])
                if dead & set(after.agent_ids):
                    found.append(f"{history.name}@{index}: weight left on {dead}")
                reached = index + 1
        except (fs.NonFiniteAllocationInput, fs.UnusableLiving) as refused:
            # A refusal ends this walk only; what was found before it stands.
            found.append(f"{history.name}@{reached}: refused: {refused}")
    assert not found, "\n".join(found)


# --- EV-E20 (generated, authorized by Task V5.2) --------------------------------


@dataclass(frozen=True)
class _Regime:
    """One regime of a declared stream that ``H`` is cited for."""

    stream: str
    best: str
    first: int
    last: int
    horizon: int


def _horizon(contract: Mapping[str, object], P: int, gap: float) -> int:
    """§2.10(1): ``H = floor(ln(1 + (1 - s) P / s) / (eta Delta)) + 1``."""
    eta = float(contract["eta"])  # ty: ignore[invalid-argument-type]
    s = float(contract["share_rate"])  # ty: ignore[invalid-argument-type]
    return math.floor(math.log(1.0 + (1.0 - s) * P / s) / (eta * gap)) + 1


def _manifest() -> dict:
    data = (FAMILY / "manifest.toml").read_bytes()
    assert hashlib.sha256(data).hexdigest() == MANIFEST_SHA256, (
        "E's manifest is not the one pinned at MANIFEST_SHA256"
    )
    manifest = tomllib.loads(data.decode("utf-8"))
    assert manifest["authorized_by"] == generated.AUTHORIZED_BY
    assert manifest["generator_version"] == generated.GENERATOR_VERSION
    return manifest


def _declared_regimes(
    contract: Mapping[str, object], entry: Mapping[str, object], warmup: int
) -> list[_Regime]:
    """The regimes ``H`` is cited for: pathwise streams only (§2.10(3)), after the warm-up.

    Decision points run from 1 to ``T - 1``; the first with a defined loss is
    ``warmup + 1``. A switch stream's cited regime starts at its switch; the
    no-switch control's at the first defined point.
    """
    if entry["kind"] == "all_noise" or entry["noise"] != "pathwise":
        return []
    T = int(entry["T"])  # ty: ignore[invalid-argument-type]
    P = int(entry["K"])  # ty: ignore[invalid-argument-type]
    H = _horizon(contract, P, float(entry["regime_gap"]))  # ty: ignore[invalid-argument-type]
    if entry["kind"] == "switch":
        first, best = int(entry["switch_at"]), str(entry["best_after"])  # ty: ignore[invalid-argument-type]
    else:
        first, best = warmup + 1, str(entry["best_before"])
    return [_Regime(str(entry["id"]), best, first, T - 1, H)]


def _self_check(entry: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """The stream's own check: its bytes, seed, switch point and gap match its declaration."""
    data = (FAMILY / str(entry["file"])).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["id"]
    header = data.split(b"\n", 2)[1].decode("ascii")
    assert f"seed={entry['seed']}" in header, entry["id"]
    reference, outcomes = generated.parse(data)
    if entry["kind"] == "all_noise" or entry["noise"] != "pathwise":
        return reference, outcomes
    names = generated.expert_ids(int(entry["K"]))  # ty: ignore[invalid-argument-type]
    sigma = float(entry["sigma"])  # ty: ignore[invalid-argument-type]
    gap = float(entry["regime_gap"]) * sigma  # ty: ignore[invalid-argument-type]
    switch = int(entry.get("switch_at", outcomes.shape[0]))  # ty: ignore[invalid-argument-type]
    for rows, best in (
        (slice(1, switch), entry["best_before"]),
        (slice(switch, None), entry.get("best_after")),
    ):
        block = outcomes[rows]
        if best is None or block.size == 0:
            continue
        b = names.index(str(best))
        others = np.delete(block, b, axis=1)
        assert np.allclose(block[:, [b]] - others, gap, rtol=0, atol=1e-12), entry["id"]
    return reference, outcomes


def _lead_breaches(fs, aggregator, entry, outcomes, reference, regimes) -> list[str]:
    names = generated.expert_ids(int(entry["K"]))
    state = aggregator.start(tuple(names))
    awake = np.ones(len(names), dtype=bool)
    leads: dict[int, bool] = {}
    best = {r.best: names.index(r.best) for r in regimes}
    for t in range(1, outcomes.shape[0]):
        state = aggregator.update(
            state,
            living=tuple(names),
            returns=outcomes[t],
            recorded=awake,
            reference=float(reference[t]),
        )
        for regime in regimes:
            if regime.first <= t <= regime.last:
                b = best[regime.best]
                rivals = np.delete(state.weights, b)
                leads[t] = bool(state.weights[b] > rivals.max())
    found = []
    for regime in regimes:
        due = regime.first + regime.horizon - 1
        missed = [t for t in range(due, regime.last + 1) if not leads[t]]
        if missed:
            found.append(
                f"{regime.stream}: {regime.best} lacks the strict lead at "
                f"{missed[:5]} (due by {due}, H = {regime.horizon})"
            )
    return found


def _print_label(entry: Mapping[str, object]) -> None:
    print(
        f"Generated (authorized by Task V5.2): two_regime_stream "
        f"{entry['generator_version']} stream {entry['id']} seed {entry['seed']}"
    )


def test_tracks_expert_switch() -> None:
    record = oracle.read_record()
    contract = record.contract
    warmup = int(contract["scale_warmup"])  # ty: ignore[invalid-argument-type]
    manifest = _manifest()
    entries = manifest["stream"]
    # H first, from the declaration alone, before any stream is read or run.
    cited = {e["id"]: _declared_regimes(contract, e, warmup) for e in entries}
    horizons = {i: [r.horizon for r in regimes] for i, regimes in cited.items()}
    assert horizons == {
        **{i: [] for i in ("E05", "E06", "E08")},
        **{i: [h] for i, h in DECLARED_HORIZONS.items()},
    }, f"the cited horizons {horizons} are not the declaration's"
    for entry in entries:
        for regime in cited[entry["id"]]:
            length = regime.last - regime.first + 1
            print(f"{regime.stream}: H = {regime.horizon}, regime updates = {length}")
            assert regime.horizon <= length, (
                f"{regime.stream}: H = {regime.horizon} exceeds the regime's "
                f"{length} updates; route to a spawned Oracle, never re-set eta, "
                "share_rate or E"
            )
        if not cited[entry["id"]]:
            print(f"{entry['id']}: condition C is not pathwise; no horizon claimed")
    checked = [(entry, *_self_check(entry)) for entry in entries]
    fs = _module()
    aggregator = fs.FixedShare(
        contract_sha256=record.sha256, eta=fs.ETA, share_rate=fs.SHARE_RATE
    )
    found: list[str] = []
    for entry, reference, outcomes in checked:
        _print_label(entry)
        if cited[entry["id"]]:
            found += _lead_breaches(
                fs, aggregator, entry, outcomes, reference, cited[entry["id"]]
            )
    assert not found, "\n".join(found)


def test_a_lead_arriving_after_the_horizon_is_a_breach() -> None:
    record = oracle.read_record()
    contract = record.contract
    warmup = int(contract["scale_warmup"])  # ty: ignore[invalid-argument-type]
    entry = next(e for e in _manifest()["stream"] if e["id"] == "E01")
    assert entry["kind"] == "switch", "E01 is not the switch stream it was declared"
    (regime,) = _declared_regimes(contract, entry, warmup)
    assert regime.first == int(entry["switch_at"])
    reference, outcomes = _self_check(entry)
    fs = _module()
    aggregator = fs.FixedShare(
        contract_sha256=record.sha256, eta=fs.ETA, share_rate=fs.SHARE_RATE
    )
    _print_label(entry)
    # The premise: at the switch itself the switched-to expert has no strict lead.
    names = generated.expert_ids(int(entry["K"]))
    state = aggregator.start(tuple(names))
    for t in range(1, regime.first + 1):
        state = aggregator.update(
            state,
            living=tuple(names),
            returns=outcomes[t],
            recorded=np.ones(len(names), dtype=bool),
            reference=float(reference[t]),
        )
    b = names.index(regime.best)
    led = bool(state.weights[b] > np.delete(state.weights, b).max())
    print(f"premise: {regime.best} holds a strict lead at switch_at: {led}")
    assert not led, "E01's switched-to expert already leads at the switch"
    found = _lead_breaches(
        fs,
        aggregator,
        entry,
        outcomes,
        reference,
        [replace(regime, horizon=1)],
    )
    assert found, "a lead arriving after a horizon of one was not a breach"
