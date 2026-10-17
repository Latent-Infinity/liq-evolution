"""Every use of generated data in this test tree carries its authorization.

Generated (authorized by Task V5.2) —
``liq-docs/plans/oracle-v5-generated-streams-2026-10-15.md`` §8(4). The check
runs in the default suite and asserts three things over ``tests/``:

(a) every file that contains ``fixtures/generated`` or imports
    ``tests.support.two_regime_stream`` also contains the literal
    ``authorized by Task V5.2`` (this module and ``__pycache__`` excluded);
(b) every file under ``tests/fixtures/generated/ecology/two_regime/`` has a
    manifest entry whose sha256 matches, every entry names a file that exists,
    and the manifest's ``authorized_by`` is the authorization string;
(c) no path contains an ``approved`` directory, and no generated stream file
    (one whose first line is the generator's label) sits outside
    ``tests/fixtures/generated/``.

The cases below run the check on the real tree, and on ``tmp_path`` copies
that break each rule, so the check is shown to bite without editing any
repository file.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tomllib
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
TOKEN = b"authorized by Task V5.2"
AUTHORIZED_BY = "Task V5.2 / liq-docs/plans/oracle-v5-generated-streams-2026-10-15.md"
GENERATED_DIR = Path("fixtures/generated")
FAMILY_DIR = GENERATED_DIR / "ecology/two_regime"
STREAM_LABEL = b"# Generated (authorized by Task V5.2) by two_regime_stream"
_REFERENCE = re.compile(
    rb"fixtures/generated"
    rb"|tests\.support\.two_regime_stream"
    rb"|from\s+tests\.support\s+import\s+[^\n]*\btwo_regime_stream\b"
)
_THIS = Path(__file__).name


def _files(tests_root: Path) -> list[Path]:
    return sorted(
        p
        for p in tests_root.rglob("*")
        if p.is_file() and "__pycache__" not in p.relative_to(tests_root).parts
    )


def _unlabelled_references(tests_root: Path) -> list[str]:
    found = []
    for path in _files(tests_root):
        if path.name == _THIS and path.parent == tests_root / "support":
            continue
        data = path.read_bytes()
        if _REFERENCE.search(data) and TOKEN not in data:
            found.append(
                f"{path.relative_to(tests_root)}: references generated data "
                f"without {TOKEN.decode()!r}"
            )
    return found


def _manifest_mismatches(tests_root: Path) -> list[str]:
    family = tests_root / FAMILY_DIR
    manifest_path = family / "manifest.toml"
    if not manifest_path.is_file():
        return [f"{FAMILY_DIR / 'manifest.toml'}: missing"]
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    found = []
    if manifest.get("authorized_by") != AUTHORIZED_BY:
        found.append(f"{FAMILY_DIR / 'manifest.toml'}: authorized_by is not V5.2's")
    recorded = {e["file"]: e["sha256"] for e in manifest.get("stream", [])}
    on_disk = {p.name for p in family.iterdir() if p.is_file()} - {"manifest.toml"}
    for name in sorted(on_disk):
        digest = hashlib.sha256((family / name).read_bytes()).hexdigest()
        if recorded.get(name) != digest:
            found.append(f"{FAMILY_DIR / name}: no manifest entry with its sha256")
    for name in sorted(set(recorded) - on_disk):
        found.append(f"{FAMILY_DIR / name}: in the manifest but not on disk")
    return found


def _misplaced_paths(tests_root: Path) -> list[str]:
    found = []
    for path in _files(tests_root):
        rel = path.relative_to(tests_root)
        if "approved" in rel.parts[:-1]:
            found.append(f"{rel}: under an approved/ directory")
        inside = rel.parts[: len(GENERATED_DIR.parts)] == GENERATED_DIR.parts
        if not inside:
            with path.open("rb") as handle:
                if handle.readline().startswith(STREAM_LABEL):
                    found.append(f"{rel}: generated stream outside {GENERATED_DIR}/")
    return found


def label_violations(tests_root: Path) -> list[str]:
    """Every breach of rules (a), (b) and (c) under ``tests_root``."""
    return [
        *_unlabelled_references(tests_root),
        *_manifest_mismatches(tests_root),
        *_misplaced_paths(tests_root),
    ]


def assert_labels_hold(tests_root: Path) -> None:
    violations = label_violations(tests_root)
    assert not violations, "generated-data label check:\n" + "\n".join(violations)


def test_generated_data_is_labelled_in_this_tree() -> None:
    assert_labels_hold(TESTS_ROOT)


# --- the check bites, on tmp_path copies --------------------------------------


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "tests"
    shutil.copytree(TESTS_ROOT / FAMILY_DIR, root / FAMILY_DIR)
    (root / "support").mkdir()
    shutil.copy2(TESTS_ROOT / "support/two_regime_stream.py", root / "support")
    assert label_violations(root) == []
    return root


def test_unlabelled_fact_reference_is_named(tree: Path) -> None:
    fact = tree / "facts/test_allocation.py"
    fact.parent.mkdir()
    fact.write_text('FAMILY = "tests/fixtures/generated/ecology/two_regime"\n')
    violations = label_violations(tree)
    assert len(violations) == 1
    assert violations[0].startswith("facts/test_allocation.py:")
    with pytest.raises(AssertionError, match="facts/test_allocation.py"):
        assert_labels_hold(tree)


@pytest.mark.parametrize(
    "line",
    [
        "from tests.support import two_regime_stream\n",
        "from tests.support import level_shift_stream, two_regime_stream\n",
        "import tests.support.two_regime_stream as trs\n",
    ],
)
def test_unlabelled_import_is_named(tree: Path, line: str) -> None:
    (tree / "support/test_consumer.py").write_text(line)
    assert label_violations(tree) == [
        "support/test_consumer.py: references generated data without "
        "'authorized by Task V5.2'"
    ]


def test_labelled_reference_is_admitted(tree: Path) -> None:
    (tree / "support/test_consumer.py").write_text(
        "# Generated (authorized by Task V5.2)\n"
        "from tests.support import two_regime_stream\n"
    )
    assert label_violations(tree) == []


def test_edited_stream_is_named(tree: Path) -> None:
    stream = tree / FAMILY_DIR / "E01.csv"
    stream.write_bytes(stream.read_bytes() + b"\n")
    assert label_violations(tree) == [
        "fixtures/generated/ecology/two_regime/E01.csv: "
        "no manifest entry with its sha256"
    ]


def test_unlisted_and_missing_streams_are_named(tree: Path) -> None:
    (tree / FAMILY_DIR / "E01.csv").rename(tree / FAMILY_DIR / "E99.csv")
    assert label_violations(tree) == [
        "fixtures/generated/ecology/two_regime/E99.csv: "
        "no manifest entry with its sha256",
        "fixtures/generated/ecology/two_regime/E01.csv: "
        "in the manifest but not on disk",
    ]


def test_wrong_authorization_is_named(tree: Path) -> None:
    manifest = tree / FAMILY_DIR / "manifest.toml"
    manifest.write_text(
        manifest.read_text().replace(AUTHORIZED_BY, "someone else"),
    )
    assert label_violations(tree) == [
        "fixtures/generated/ecology/two_regime/manifest.toml: "
        "authorized_by is not V5.2's"
    ]


def test_missing_manifest_is_named(tree: Path) -> None:
    shutil.rmtree(tree / FAMILY_DIR)
    assert label_violations(tree) == [
        "fixtures/generated/ecology/two_regime/manifest.toml: missing"
    ]


def test_approved_directory_is_named(tree: Path) -> None:
    approved = tree / "fixtures/approved/ecology/bars.csv"
    approved.parent.mkdir(parents=True)
    approved.write_text("x\n")
    assert label_violations(tree) == [
        "fixtures/approved/ecology/bars.csv: under an approved/ directory"
    ]


def test_stream_outside_generated_is_named(tree: Path) -> None:
    shutil.copy2(tree / FAMILY_DIR / "E01.csv", tree / "support/E01.csv")
    assert label_violations(tree) == [
        "support/E01.csv: generated stream outside fixtures/generated/"
    ]
