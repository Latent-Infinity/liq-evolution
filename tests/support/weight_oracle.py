"""An independent rebuild of the fixed-share weights, from the frozen record's text.

The record is ``liq-docs/plans/agent-ecology-aggregator-contract.md``, read
here only at the sha256 its freeze names. Every formula below is written from
that record's §2.2–§2.7 and nothing else: it is never derived from, and never
imports, the implementation it checks. It is deliberately written another way
— pure Python over mappings keyed by agent, ``math.fsum`` sums and a
max-shifted logsumexp — so an error shared by both would have to be made twice.

**The comparison rule** (the record's §7, freeze record §4(3)): a rebuilt
weight matches a printed or implemented one when it lies within
``16 * P * 2**-52`` of it, the §2.6(4) bound; a newborn's weight and the floor
``s / P`` are compared exactly, with no tolerance (§2.6(3), §2.7(2)).
:func:`self_check` states that rule and applies it to the record's three
worked cases.

Nothing here is market data. The worked cases are the record's own hand-set
inputs.
"""

from __future__ import annotations

import hashlib
import math
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]

#: The frozen aggregator contract record, as the documentation repository holds it.
RECORD = (
    REPOSITORY.parent / "liq-docs" / "plans" / "agent-ecology-aggregator-contract.md"
)

#: The sha256 the record was frozen at
#: (``liq-docs/plans/oracle-v5-aggregator-contract-frozen-2026-10-15.md``).
FROZEN_SHA256 = "e25af4f6069ace503ff8c2d97b6ecd8e8676ed838032c949262f0cb9032da0bb"

#: The comparison rule, stated as the record's §7 states it.
COMPARISON_RULE = (
    "a rebuilt weight matches when it lies within 16*P*2**-52 of the other; "
    "a newborn's weight and the floor s/P are compared exactly"
)

_TOML_BLOCK = re.compile(r"```toml\n(.*?)\n```", re.DOTALL)


@dataclass(frozen=True)
class Record:
    """The frozen record: its bytes, its sha256 and its machine-readable section."""

    text: str
    sha256: str
    fields: Mapping[str, object]

    @property
    def contract(self) -> Mapping[str, object]:
        """The ``[contract]`` table."""
        return self.fields["contract"]  # ty: ignore[invalid-return-type]


def read_record(path: Path = RECORD) -> Record:
    """Read the record and refuse any bytes other than the frozen ones."""
    assert path.is_file(), (
        f"{path} is missing; the aggregator checks read the frozen contract "
        "record from the sibling liq-docs repository"
    )
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == FROZEN_SHA256, (
        f"{path} hashes to {digest}, not the frozen {FROZEN_SHA256}; a record "
        "of other bytes is a new record, not a reading error"
    )
    text = data.decode("utf-8")
    blocks = _TOML_BLOCK.findall(text)
    assert len(blocks) == 1, "the record carries exactly one toml block"
    return Record(text=text, sha256=digest, fields=tomllib.loads(blocks[0]))


def tolerance(living_count: int) -> float:
    """The §2.6(4) sum and comparison bound at ``living_count`` agents."""
    return 16 * living_count * 2.0**-52


# --- the update, rebuilt from §2.2–§2.7 -----------------------------------------


def loss_step_and_mix(
    before: Mapping[str, float],
    living: Sequence[str],
    losses: Mapping[str, float],
    *,
    eta: float,
    share_rate: float,
) -> dict[str, float]:
    """One update over ``living`` from ``before``, with scaled ``losses`` for the awake.

    ``losses`` names exactly the awake survivors and their scaled loss; an
    agent of ``living`` absent from ``before`` is a newborn.
    """
    survivors = [a for a in sorted(before) if a in set(living)]
    awake = [a for a in survivors if a in losses]
    u = {a: before[a] for a in survivors}
    if awake:
        # A zero prior weight is outside the contract (the floor forbids it) but
        # is rebuilt as -inf rather than crashing, so a vector that breached the
        # floor at the previous point is reported here, not hidden by an error.
        logs = {
            a: (math.log(before[a]) if before[a] > 0.0 else -math.inf) - eta * losses[a]
            for a in awake
        }
        top = max(logs.values())
        lse = top + math.log(math.fsum(math.exp(x - top) for x in logs.values()))
        mass = math.fsum(before[a] for a in awake)
        for a in awake:
            u[a] = math.exp(logs[a] - lse) * mass
    total = math.fsum(u.values())
    v = {a: u.get(a, 0.0) / total if a in u else 0.0 for a in living}
    count = len(living)
    return {a: (1.0 - share_rate) * v[a] + share_rate / count for a in sorted(living)}


@dataclass(frozen=True)
class Scale:
    """The reference scale's whole state: how many returns, and their squares' sum."""

    count: int = 0
    sum_of_squares: float = 0.0

    def defined(self, warmup: int) -> bool:
        """Whether the loss is defined at the next decision point (§2.2(3))."""
        return self.count >= warmup and self.sum_of_squares > 0.0

    def sigma(self) -> float:
        """The reference scale σ̂ (§2.2(2))."""
        return math.sqrt(self.sum_of_squares / self.count)

    def after(self, reference: float | None) -> Scale:
        """The scale once the reference return recorded at this point is added."""
        if reference is None:
            return self
        return Scale(self.count + 1, self.sum_of_squares + reference * reference)


def update(
    before: Mapping[str, float],
    living: Sequence[str],
    returns: Mapping[str, float],
    scale: Scale,
    reference: float | None,
    *,
    eta: float,
    share_rate: float,
    warmup: int,
) -> tuple[dict[str, float], Scale]:
    """One decision point: losses from raw ``returns``, then the update, then the scale.

    ``returns`` names every agent with a recorded outcome at this point; only
    survivors among them are awake, and only once the loss is defined.
    """
    losses: dict[str, float] = {}
    if scale.defined(warmup):
        sigma = scale.sigma()
        losses = {a: -r / sigma for a, r in returns.items() if a in before}
    weights = loss_step_and_mix(before, living, losses, eta=eta, share_rate=share_rate)
    return weights, scale.after(reference)


# --- the record's worked cases (§7) ---------------------------------------------


@dataclass(frozen=True)
class WorkedCase:
    """One of §7's cases: the hand-set inputs, and the printed weights and floor."""

    number: int
    before: Mapping[str, float]
    living: tuple[str, ...]
    losses: Mapping[str, float]
    newborns: frozenset[str]
    printed: Mapping[str, float]
    floor: float


#: §7's hand-set inputs, transcribed from each case's own sentence.
_INPUTS = {
    1: ({"a1": 1 / 3, "a2": 1 / 3, "a3": 1 / 3}, ("a1", "a2", "a3"), {}),
    2: (
        {"a1": 0.5, "a2": 0.3, "a3": 0.2},
        ("a1", "a2", "a4"),
        {"a1": 0.5, "a2": -1.0},
    ),
    3: ({"a1": 0.6, "a2": 0.4}, ("a1", "a2"), {}),
}

_CASE = re.compile(r"^(\d)\. \*\*", re.MULTILINE)
_WEIGHT = re.compile(r"w\((a\d)\) = `([^`]+)`")
_FLOOR = re.compile(r"floor s/Pₜ = `([^`]+)`")


def worked_cases(record: Record) -> tuple[WorkedCase, ...]:
    """§7's three cases, their printed values read from the record's own text."""
    section = record.text.split("## 7. Worked cases", 1)[1].split("## 8.", 1)[0]
    starts = [(int(m.group(1)), m.start()) for m in _CASE.finditer(section)]
    cases = []
    for index, (number, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(section)
        body = section[start:end]
        before, living, losses = _INPUTS[number]
        floor = _FLOOR.search(body)
        assert floor is not None, f"case {number} prints no floor"
        cases.append(
            WorkedCase(
                number=number,
                before=before,
                living=living,
                losses=losses,
                newborns=frozenset(set(living) - set(before)),
                printed={a: float(v) for a, v in _WEIGHT.findall(body)},
                floor=float(floor.group(1)),
            )
        )
    return tuple(cases)


def mismatches(
    rebuilt: Mapping[str, float],
    other: Mapping[str, float],
    *,
    newborns: frozenset[str],
    floor: float | None = None,
    share_rate: float,
) -> list[str]:
    """Every way ``other`` fails the comparison rule against ``rebuilt``."""
    found = []
    if set(rebuilt) != set(other):
        return [f"agents differ: {sorted(rebuilt)} against {sorted(other)}"]
    count = len(rebuilt)
    bound = tolerance(count)
    if floor is not None and floor != share_rate / count:
        found.append(f"floor {floor!r} is not s/P = {share_rate / count!r}")
    for agent_id in sorted(rebuilt):
        mine, theirs = rebuilt[agent_id], other[agent_id]
        if agent_id in newborns:
            if theirs != share_rate / count:
                found.append(f"newborn {agent_id}: {theirs!r} is not s/P exactly")
        elif abs(mine - theirs) > bound:
            found.append(f"{agent_id}: {theirs!r} against {mine!r} beyond {bound!r}")
    return found


def self_check(record: Record) -> list[str]:
    """Rebuild §7's cases and compare them under :data:`COMPARISON_RULE`."""
    contract = record.contract
    eta = float(contract["eta"])  # ty: ignore[invalid-argument-type]
    share_rate = float(contract["share_rate"])  # ty: ignore[invalid-argument-type]
    found = []
    for case in worked_cases(record):
        rebuilt = loss_step_and_mix(
            case.before, case.living, case.losses, eta=eta, share_rate=share_rate
        )
        found += [
            f"case {case.number}: {each}"
            for each in mismatches(
                rebuilt,
                case.printed,
                newborns=case.newborns,
                floor=case.floor,
                share_rate=share_rate,
            )
        ]
    return found
