"""The two-regime loss stream: its declared arithmetic, and the family on disk.

Generated (authorized by Task V5.2) —
``liq-docs/plans/oracle-v5-generated-streams-2026-10-15.md``. Nothing here is
market data: every number is a dimensionless per-expert outcome or reference
return produced by the declared arithmetic in
``two-regime-generator-spec.md``.

Two kinds of case live here. The arithmetic cases hold the generator to the
spec's text — its draw order, its operation order and its seed-range refusal.
The regeneration cases hold the materialised evidence family under
``tests/fixtures/generated/ecology/two_regime/`` to its manifest: every stream
regenerates byte-identically from the seed and parameters the manifest records,
and the manifest names the module that produced it.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

import numpy as np
import pytest

from tests.support import two_regime_stream as trs

FAMILY_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures/generated/ecology/two_regime"
)
MANIFEST = FAMILY_DIR / "manifest.toml"
MODULE = Path(trs.__file__).resolve()

#: The evidence-family declaration's sha256, as ``SHA256SUMS`` beside it records.
EVIDENCE_DECLARATION_SHA256 = (
    "039c1f629f21c604239f12238399936bc587b47c0e71ece60de7ef5e029eb458"
)
GENERATOR_SPEC_SHA256 = (
    "8a6777ab2e7b2f65a461d732f60fdb0be673ccc50f40bb090cc5543bc0a6c877"
)

E_SEED = 1_000_000
C_SEED = 0


def _spec(**overrides: object) -> trs.StreamSpec:
    fields: dict[str, object] = {
        "seed": E_SEED,
        "P": 4,
        "T": 12,
        "kind": "switch",
        "noise": "pathwise",
        "switch_at": 5,
        "Delta": 0.5,
        "sigma": 0.01,
        "lam": 0.5,
    }
    fields.update(overrides)
    return trs.StreamSpec(**fields)  # type: ignore[arg-type]


def _manifest() -> dict:
    return tomllib.loads(MANIFEST.read_text(encoding="utf-8"))


# --- identity -----------------------------------------------------------------


def test_module_declares_version_and_authorization() -> None:
    assert isinstance(trs.GENERATOR_VERSION, str) and trs.GENERATOR_VERSION
    assert trs.AUTHORIZED_BY == (
        "Task V5.2 / liq-docs/plans/oracle-v5-generated-streams-2026-10-15.md"
    )


def test_module_imports_no_data_store() -> None:
    source = MODULE.read_text(encoding="utf-8")
    for forbidden in ("liq.data", "liq.store", "adapters"):
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source


def test_expert_ids_sort_in_index_order() -> None:
    ids = trs.expert_ids(12)
    assert ids[0] == "e00000" and ids[11] == "e00011"
    assert sorted(ids) == ids


# --- seed-range refusal -------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "seed"),
    [("E", 999_999), ("E", 2_000_000), ("C", 1_000_000), ("C", -1)],
)
def test_seed_outside_family_range_is_refused(family: str, seed: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        trs.generate(_spec(seed=seed), family=family)


def test_unknown_family_is_refused() -> None:
    with pytest.raises(ValueError, match="family"):
        trs.generate(_spec(), family="X")


@pytest.mark.parametrize(("family", "seed"), [("E", 1_999_999), ("C", 999_999)])
def test_range_bounds_are_inclusive(family: str, seed: int) -> None:
    trs.generate(_spec(seed=seed), family=family)


def test_reference_follows_the_spec_in_either_family() -> None:
    e = trs.generate(_spec(seed=E_SEED), family="E")
    replay = np.random.default_rng(E_SEED)
    z = replay.integers(0, 2, size=12).astype(np.float64) * 2.0 - 1.0
    np.testing.assert_array_equal(e.reference, 0.01 * z)
    c = trs.generate(_spec(seed=C_SEED), family="C")
    replay = np.random.default_rng(C_SEED)
    z = replay.integers(0, 2, size=12).astype(np.float64) * 2.0 - 1.0
    np.testing.assert_array_equal(c.reference, 0.01 * z)


# --- parameter validation -----------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"P": 1},
        {"T": 1},
        {"kind": "three_regime"},
        {"noise": "brownian"},
        {"noise": None},
        {"switch_at": 1},
        {"switch_at": 12},
        {"switch_at": None},
        {"Delta": 0.0},
        {"Delta": None},
        {"sigma": 0.0},
        {"lam": -0.1},
    ],
)
def test_invalid_parameters_are_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        trs.generate(_spec(**overrides), family="E")


def test_all_noise_needs_no_gap_noise_or_switch() -> None:
    stream = trs.generate(
        _spec(kind="all_noise", noise=None, switch_at=None, Delta=None), family="E"
    )
    assert stream.best is None


# --- the arithmetic, replayed from the spec's text ---------------------------


def _replay(seed: int, P: int, T: int, need_eps: bool):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=T).astype(np.float64) * 2.0 - 1.0
    perm = rng.permutation(P)
    eps = rng.standard_normal((T, P)) if need_eps else None
    return z, int(perm[0]), int(perm[1]), eps


def test_switch_pathwise_matches_spec() -> None:
    s = trs.generate(_spec(), family="E")
    z, b1, b2, _ = _replay(E_SEED, 4, 12, need_eps=False)
    assert (b1, b2) == (s.B1, s.B2)
    assert b1 != b2
    delta = 0.5 * 0.01
    expected = np.full((12, 4), -delta)
    for t in range(12):
        expected[t, b1 if t < 5 else b2] = 0.0
    expected = 0.01 * z[:, None] + expected
    np.testing.assert_array_equal(s.outcomes, expected)
    assert s.best is not None
    assert list(s.best) == [b1] * 5 + [b2] * 7
    # Pathwise: the best expert beats every rival by delta (to rounding).
    for t in range(12):
        gaps = s.outcomes[t, s.best[t]] - np.delete(s.outcomes[t], s.best[t])
        np.testing.assert_allclose(gaps, delta, rtol=1e-12)


def test_no_switch_keeps_one_best_expert() -> None:
    s = trs.generate(_spec(kind="no_switch", switch_at=None), family="E")
    assert s.best is not None
    assert set(s.best.tolist()) == {s.B1}


def test_idiosyncratic_adds_scaled_third_draw() -> None:
    s = trs.generate(_spec(noise="idiosyncratic"), family="E")
    z, b1, b2, eps = _replay(E_SEED, 4, 12, need_eps=True)
    assert eps is not None
    drift = np.full((12, 4), -(0.5 * 0.01))
    for t in range(12):
        drift[t, b1 if t < 5 else b2] = 0.0
    expected = (0.01 * z[:, None] + drift) + (0.5 * 0.01) * eps
    np.testing.assert_array_equal(s.outcomes, expected)


def test_all_noise_is_scaled_standard_normal() -> None:
    s = trs.generate(
        _spec(kind="all_noise", noise=None, switch_at=None, Delta=None), family="E"
    )
    z, _, _, eps = _replay(E_SEED, 4, 12, need_eps=True)
    np.testing.assert_array_equal(s.outcomes, 0.01 * eps)
    np.testing.assert_array_equal(s.reference, 0.01 * z)


def test_generation_is_deterministic() -> None:
    a = trs.render(trs.generate(_spec(), family="E"), stream_id="X")
    b = trs.render(trs.generate(_spec(), family="E"), stream_id="X")
    assert a == b


def test_render_round_trips_exactly() -> None:
    s = trs.generate(_spec(noise="idiosyncratic"), family="E")
    reference, outcomes = trs.parse(trs.render(s, stream_id="X"))
    np.testing.assert_array_equal(reference, s.reference)
    np.testing.assert_array_equal(outcomes, s.outcomes)


def test_rendered_stream_is_labelled_generated() -> None:
    text = trs.render(trs.generate(_spec(), family="E"), stream_id="X").decode()
    assert text.startswith(trs.STREAM_HEADER_PREFIX)
    assert "authorized by Task V5.2" in text.splitlines()[0]
    assert trs.GENERATOR_VERSION in text.splitlines()[0]


# --- the materialised evidence family -----------------------------------------


def test_manifest_labels_the_family() -> None:
    m = _manifest()
    assert m["authorized_by"] == trs.AUTHORIZED_BY
    assert m["signal_bearing"] is False
    assert m["citable"] is False
    assert m["family"] == "E"
    assert m["declaration_sha256"] == EVIDENCE_DECLARATION_SHA256
    assert m["generator_spec_sha256"] == GENERATOR_SPEC_SHA256


def test_manifest_names_the_module_that_wrote_it() -> None:
    m = _manifest()
    module_sha = hashlib.sha256(MODULE.read_bytes()).hexdigest()
    assert m["generator_version"] == trs.GENERATOR_VERSION
    assert m["generator_module_sha256"] == module_sha
    for entry in m["stream"]:
        assert entry["generator_version"] == trs.GENERATOR_VERSION
        assert entry["generator_module_sha256"] == module_sha


def test_family_holds_every_declared_stream() -> None:
    ids = [entry["id"] for entry in _manifest()["stream"]]
    assert ids == [f"E{i:02d}" for i in range(1, 9)]
    kinds = {entry["kind"] for entry in _manifest()["stream"]}
    assert kinds == {"switch", "no_switch", "all_noise"}


def test_family_stays_within_one_megabyte() -> None:
    total = sum(p.stat().st_size for p in FAMILY_DIR.iterdir() if p.is_file())
    assert total <= 1_000_000


def _entries() -> list[dict]:
    return list(_manifest()["stream"])


@pytest.mark.parametrize("entry", _entries(), ids=lambda e: e["id"])
def test_stream_regenerates_byte_identically(entry: dict) -> None:
    stream = trs.generate(trs.spec_from_manifest(entry), family="E")
    regenerated = trs.render(stream, stream_id=entry["id"])
    committed = (FAMILY_DIR / entry["file"]).read_bytes()
    print(
        f"{entry['id']}: Generated (authorized by Task V5.2) seed={entry['seed']} "
        f"generator_version={trs.GENERATOR_VERSION}"
    )
    assert regenerated == committed
    assert hashlib.sha256(committed).hexdigest() == entry["sha256"]
