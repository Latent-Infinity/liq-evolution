"""The shipped descriptors pinned to the descriptor-ablation harness, on every behavioural axis.

The descriptor-parameter rule's Q-5 conformance clause: run over the harness's
recorded conformance sample — seed 4000, N in {24, 64}, streams S1, S3 and S4,
every reading's input windows — the shipped
:mod:`liq.evolution.ecology.descriptors` returns, on every agent and reading,
the harness's status and a normalised value within ``1e-9`` absolute of the
harness's. It is held in two halves, against the two runs whose harness
computes what the module ships:

* **Horizon and turnover**, against the ablation's first run, as the
  successor-2 rule's R2-8 narrows the clause
  (``plans/agent-ecology-adr-descriptor-parameters.successor-2-rule.md``).
  That run's harness computes the retired intercept slope for beta, so its
  beta is neither compared nor read.
* **Beta**, against the ablation's second run, whose harness computes the
  shipped estimator — the least-squares slope through the origin, under
  ``ecology-descriptors-2``
  (``plans/agent-ecology-adr-descriptor-parameters.successor-3-rule.md`` S3-4).
  The raw estimate the harness records is held to the module's raw value too.

**What the harness's values are.** Read, never recomputed here, from each run
as ``liq-docs`` holds it: the pin samples (``main/pin-sample-<stream>-N<N>.npz``)
and the harness's own per-reading output for that seed
(``main/main-<stream>-N<N>-b0.npz``, ``s4000/*``). Every file read is first held
to the run's ``SHA256SUMS``, and that list to the hash its rule's integrity
check or the ADR records; what is parsed is the bytes that were hashed. The
second run's N = 64 outputs are over 1 MB and so are untracked by rule: they
are read from the working tree and admitted only by that checksum. A second
case per half holds the recorded values to the run's tracked harness formula
(``harness-src/desc.py``), so the pin compares against the harness's
arithmetic and not merely a file.

**Nothing is written beside the harness, and nothing beside it is read.** Its
formula is compiled from the verified bytes and executed into a fresh module,
never imported through the import system: an import would write a bytecode
cache into the artifact tree, and would execute any matching cache found
there in place of the hashed source.

**Describedness is the harness's.** The sample holds no window for an agent
the harness found undescribed, so such an agent is given no outcomes here and
is checked only to be reported undescribed; whether the module decides
describedness as the harness does is not something this sample can show.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from liq.evolution.ecology import descriptors

REPOSITORY = Path(__file__).resolve().parents[2]

#: The descriptor ablation's runs, as the documentation repository holds them.
ABLATION = (
    REPOSITORY.parent
    / "liq-docs"
    / "artifacts"
    / "agent-ecology"
    / "descriptor-ablation"
)


@dataclass(frozen=True)
class _Run:
    """One ablation run's directory and the recorded hash of its checksum list."""

    root: Path
    sums_sha256: str


#: The first run; its checksum list's hash as the successor-2 rule's
#: integrity check records it.
RUN_1 = _Run(
    ABLATION / "run-1",
    "d84368ff3098833fdfa3ba6b03235c11d3e1117ddd67246b78cca6c9cc772fd6",
)

#: The second run; its checksum list's hash as the ADR records the run.
RUN_2 = _Run(
    ABLATION / "run-2",
    "2567d2065e319ebe5fee00f5f3c3880a27e0f7993487a7d5771f31b6ada6cba2",
)

RUNS = (RUN_1, RUN_2)

#: The recorded conformance sample (successor rule Q-15).
SEED = 4000
STREAMS = ("S1", "S3", "S4")
POPULATIONS = (24, 64)

#: The pin's tolerance on a normalised value (successor rule Q-5).
TOLERANCE = 1e-9

#: The harness's status codes (``harness-src/desc.py``).
OK, EXCLUDED_NOT_FINITE, EXCLUDED_OUT_OF_RANGE, UNDESCRIBED = 0, 1, 2, -1

#: The shipped module's exclusion reasons as the harness codes them on horizon
#: and turnover. A reason the harness has no code for maps to one no harness
#: status can equal.
_CODES: Mapping[descriptors.Exclusion | None, int] = {
    None: OK,
    descriptors.Exclusion.NOT_FINITE: EXCLUDED_NOT_FINITE,
    descriptors.Exclusion.OUTSIDE_UNIT_INTERVAL: EXCLUDED_OUT_OF_RANGE,
}
#: On beta the harness codes an undefined estimate as not finite (S3-4: an
#: undefined beta "is NaN, raw and normalised, status excluded_nonfinite").
_BETA_CODES: Mapping[descriptors.Exclusion | None, int] = {
    **_CODES,
    descriptors.Exclusion.UNDEFINED: EXCLUDED_NOT_FINITE,
}
_NO_HARNESS_CODE = 99


@dataclass(frozen=True)
class _Axis:
    """A pinned axis, its column in the harness's ``norm`` and ``status``, its codes."""

    name: str
    column: int
    codes: Mapping[descriptors.Exclusion | None, int]


HORIZON_AND_TURNOVER = (
    _Axis(descriptors.HOLDING_HORIZON, 0, _CODES),
    _Axis(descriptors.TURNOVER, 1, _CODES),
)
BETA = (_Axis(descriptors.BETA, 2, _BETA_CODES),)


def _listed(run: _Run, relative: str) -> str:
    """The sha256 the run's checksum list names for one file, the list itself held first."""
    sums = run.root / "SHA256SUMS"
    assert sums.is_file(), (
        f"{run.root} is missing; the conformance pin reads the ablation's "
        "outputs from the sibling liq-docs repository"
    )
    listing = sums.read_bytes()
    assert hashlib.sha256(listing).hexdigest() == run.sums_sha256, (
        f"{sums} is not the checksum list the run's record names"
    )
    listed = {
        name: digest
        for digest, name in (
            line.split(maxsplit=1) for line in listing.decode().splitlines()
        )
    }
    return listed[f"./{relative}"]


def _read_once(path: Path, sha256: str) -> bytes:
    """A file's bytes, read once and refused unless they hash to ``sha256``."""
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == sha256, (
        f"{path} is not the file the run's checksum list names"
    )
    return content


def _verified(run: _Run, relative: str) -> bytes:
    """A run file's bytes, refused unless they are the ones its checksum list names."""
    return _read_once(run.root / relative, _listed(run, relative))


def _npz(run: _Run, relative: str) -> Mapping[str, np.ndarray]:
    return np.load(io.BytesIO(_verified(run, relative)))


@dataclass(frozen=True)
class _Harness:
    """One stream and population's recorded sample and the harness's output on it."""

    readings: np.ndarray  # (k,)
    exposures: np.ndarray  # (k, n, W)
    outcomes: np.ndarray  # (k, n, W)
    reference: np.ndarray  # (k, W)
    described: np.ndarray  # (k, n)
    norm: np.ndarray  # (k, n, 3)
    status: np.ndarray  # (k, n, 3)
    beta_raw: np.ndarray  # (k, n)


def _harness(run: _Run, stream: str, population: int) -> _Harness:
    sample = _npz(run, f"main/pin-sample-{stream}-N{population}.npz")
    output = _npz(run, f"main/main-{stream}-N{population}-b0.npz")
    readings = sample["readings"]
    assert np.array_equal(output[f"s{SEED}/readings"], readings)
    return _Harness(
        readings=readings,
        exposures=sample["x"],
        outcomes=sample["r_a"],
        reference=sample["r_ref"],
        described=output[f"s{SEED}/described"],
        norm=output[f"s{SEED}/norm"],
        status=output[f"s{SEED}/status"],
        beta_raw=output[f"s{SEED}/beta_raw"],
    )


def _settings(run: _Run) -> tuple[int, int, descriptors.Mandate]:
    """``W``, the store's capacity and the mandate, from the run's tracked config copy."""
    config = tomllib.loads(_verified(run, "harness-src/steady_state.toml").decode())
    mandate = config["mandate"]
    return (
        int(config["turnover"]["fitness_window"]),
        int(config["population"]["history_capacity"]),
        descriptors.Mandate(
            max_position=mandate["max_position"],
            max_gross_exposure=mandate["max_gross_exposure"],
            max_weight_delta=mandate["max_weight_delta"],
            max_turnover=mandate["max_turnover"],
        ),
    )


@dataclass(frozen=True)
class _SampleStore:
    """One reading's recorded windows, served as the population store serves them."""

    outcomes: Mapping[str, tuple[float, ...]]
    held: Mapping[str, tuple[float, ...]]
    reference: tuple[float, ...]

    def agent_ids(self) -> tuple[str, ...]:
        return tuple(self.outcomes)

    def history(self, agent_id: str) -> tuple[float, ...]:
        return self.outcomes[agent_id]

    def exposures(self, agent_id: str) -> tuple[float, ...]:
        return self.held[agent_id]

    def reference_returns(self) -> tuple[float, ...]:
        return self.reference


def _shipped(
    harness: _Harness,
    reading: int,
    declaration: descriptors.DescriptorDeclaration,
    axes: tuple[_Axis, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The shipped module's ``(n, len(axes))`` normalised values, raw values and codes."""
    agents = tuple(f"agent-{i}" for i in range(harness.described.shape[1]))
    described = harness.described[reading]
    store = _SampleStore(
        outcomes={
            a: tuple(harness.outcomes[reading, i]) if described[i] else ()
            for i, a in enumerate(agents)
        },
        held={
            a: tuple(harness.exposures[reading, i]) if described[i] else ()
            for i, a in enumerate(agents)
        },
        reference=tuple(harness.reference[reading]) if described.any() else (),
    )
    result = descriptors.read(store, declaration)
    values = np.full((len(agents), len(axes)), np.nan)
    raw = np.full((len(agents), len(axes)), np.nan)
    codes = np.full((len(agents), len(axes)), UNDESCRIBED, dtype=np.int64)
    for i, agent_id in enumerate(agents):
        if agent_id in result.undescribed:
            continue
        agent = result.described[agent_id]
        for column, axis in enumerate(axes):
            values[i, column] = agent.values[axis.name]
            raw[i, column] = agent.raw[axis.name]
            codes[i, column] = axis.codes.get(
                agent.excluded.get(axis.name), _NO_HARNESS_CODE
            )
    return values, raw, codes


def _assert_shipped_reproduces(
    run: _Run, stream: str, population: int, axes: tuple[_Axis, ...]
) -> None:
    """Statuses identical and normalised values within the tolerance, every reading.

    On beta the raw estimate is held to the harness's recorded ``beta_raw`` too.
    """
    window, capacity, mandate = _settings(run)
    declaration = descriptors.DescriptorDeclaration(
        window=window,
        history_capacity=capacity,
        asset_classes=("equity",),
        universe={"pin-instrument": "equity"},
        mandate=mandate,
    )
    harness = _harness(run, stream, population)
    columns = [axis.column for axis in axes]
    beta = [i for i, axis in enumerate(axes) if axis.name == descriptors.BETA]
    compared = 0
    for reading in range(len(harness.readings)):
        values, raw, codes = _shipped(harness, reading, declaration, axes)
        where = f"{run.root.name} {stream} N{population} reading {harness.readings[reading]}"
        np.testing.assert_array_equal(
            codes, harness.status[reading][:, columns], err_msg=where
        )
        described = harness.described[reading]
        np.testing.assert_allclose(
            values[described],
            harness.norm[reading][described][:, columns],
            rtol=0.0,
            atol=TOLERANCE,
            equal_nan=True,
            err_msg=where,
        )
        for column in beta:
            np.testing.assert_allclose(
                raw[described, column],
                harness.beta_raw[reading][described],
                rtol=0.0,
                atol=TOLERANCE,
                equal_nan=True,
                err_msg=f"raw beta, {where}",
            )
        compared += int(described.sum())
    # Every reading from the first a full window allows is described in full.
    assert compared == population * int((harness.readings >= window).sum())


def _load_formula(path: Path, sha256: str) -> ModuleType:
    """A harness ``desc.py``, compiled from the very bytes hashed; numpy only.

    Executed into a fresh module, never through the import system: nothing is
    registered, no bytecode cache is written beside the source, and none found
    there is read in its place.
    """
    code = compile(_read_once(path, sha256), str(path), "exec", dont_inherit=True)
    module = ModuleType("descriptor_ablation_desc")
    module.__file__ = str(path)
    exec(code, module.__dict__)
    return module


def _harness_formula(run: _Run) -> ModuleType:
    """The run's tracked ``desc.py``, as its checksum list names it."""
    relative = "harness-src/desc.py"
    return _load_formula(run.root / relative, _listed(run, relative))


def _assert_recorded_values_are_the_formula(
    run: _Run, stream: str, population: int, axes: tuple[_Axis, ...]
) -> None:
    """The run's recorded statuses and values are its tracked formula's, exactly."""
    window, _, mandate = _settings(run)
    formula = _harness_formula(run)
    harness = _harness(run, stream, population)
    columns = [axis.column for axis in axes]
    beta = any(axis.name == descriptors.BETA for axis in axes)
    for reading in range(len(harness.readings)):
        described = harness.described[reading]
        if not described.any():
            continue
        result = formula.compute(
            harness.exposures[reading][described],
            harness.outcomes[reading][described],
            harness.reference[reading],
            W=window,
            c_t=mandate.turnover_cap,
            beta_max=mandate.beta_max,
        )
        np.testing.assert_array_equal(
            result["status"][:, columns], harness.status[reading][described][:, columns]
        )
        np.testing.assert_array_equal(
            result["norm"][:, columns], harness.norm[reading][described][:, columns]
        )
        if beta:
            np.testing.assert_array_equal(
                result["beta_raw"], harness.beta_raw[reading][described]
            )


@pytest.mark.parametrize("population", POPULATIONS)
@pytest.mark.parametrize("stream", STREAMS)
def test_horizon_and_turnover_reproduce_the_harness_on_the_conformance_sample(
    stream: str, population: int
) -> None:
    _assert_shipped_reproduces(RUN_1, stream, population, HORIZON_AND_TURNOVER)


@pytest.mark.parametrize("population", POPULATIONS)
@pytest.mark.parametrize("stream", STREAMS)
def test_the_recorded_harness_values_are_its_tracked_formulas(
    stream: str, population: int
) -> None:
    _assert_recorded_values_are_the_formula(
        RUN_1, stream, population, HORIZON_AND_TURNOVER
    )


def test_the_beta_pin_is_against_the_vocabulary_the_second_run_computes() -> None:
    # A beta pin against ecology-descriptors-1 code is void (S3-4).
    assert descriptors.VOCABULARY == "ecology-descriptors-2"
    assert descriptors.BETA_ESTIMATOR == "least_squares_through_origin"


@pytest.mark.parametrize("population", POPULATIONS)
@pytest.mark.parametrize("stream", STREAMS)
def test_beta_reproduces_the_harness_on_the_conformance_sample(
    stream: str, population: int
) -> None:
    _assert_shipped_reproduces(RUN_2, stream, population, BETA)


@pytest.mark.parametrize("population", POPULATIONS)
@pytest.mark.parametrize("stream", STREAMS)
def test_the_recorded_harness_beta_is_its_tracked_formula(
    stream: str, population: int
) -> None:
    _assert_recorded_values_are_the_formula(RUN_2, stream, population, BETA)


def test_a_sample_its_checksum_list_does_not_name_is_refused(tmp_path: Path) -> None:
    copy = _Run(tmp_path / "run-2", RUN_2.sums_sha256)
    sample = "main/pin-sample-S1-N24.npz"
    for relative in (
        "SHA256SUMS",
        "harness-src/steady_state.toml",
        sample,
        "main/main-S1-N24-b0.npz",
    ):
        (copy.root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RUN_2.root / relative, copy.root / relative)
    tampered = bytearray((copy.root / sample).read_bytes())
    tampered[len(tampered) // 2] ^= 0x01
    (copy.root / sample).write_bytes(bytes(tampered))
    with pytest.raises(AssertionError, match="not the file the run's checksum list"):
        _assert_shipped_reproduces(copy, "S1", 24, BETA)


def _tree(root: Path) -> dict[str, tuple[int, str]]:
    """Every file under ``root``, ignored ones included: its size and sha256."""
    return {
        str(path.relative_to(root)): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("run", RUNS, ids=("run-1", "run-2"))
def test_loading_a_formula_writes_nothing_beside_it(
    run: _Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A fresh copy with no cache beside it, and caches allowed, so a loader
    # that writes one shows.
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    relative = "harness-src/desc.py"
    copy = tmp_path / "desc.py"
    shutil.copy2(run.root / relative, copy)
    before = _tree(tmp_path)
    _load_formula(copy, _listed(run, relative))
    assert _tree(tmp_path) == before
    assert not list(tmp_path.rglob("__pycache__"))
    assert not list(tmp_path.rglob("*.pyc"))


@pytest.mark.parametrize("run", RUNS, ids=("run-1", "run-2"))
def test_the_pin_leaves_the_run_tree_unchanged(
    run: _Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    before = _tree(run.root)
    _harness_formula(run)
    assert _tree(run.root) == before
