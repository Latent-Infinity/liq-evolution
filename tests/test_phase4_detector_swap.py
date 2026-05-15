"""Tests for Phase 4 paired detector-source evolution runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from liq.evolution.phase4_detector_swap import (
    REQUIRED_TRAINED_TERMINALS,
    run_paired_detector_swap,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _terminal_frame() -> dict[str, list[bool]]:
    return {
        "svm_regime_is_trend": [True, False, False, True, False, False],
        "svm_regime_is_range": [False, True, False, False, True, False],
        "svm_regime_is_neutral": [False, False, True, False, False, True],
        "svm_regime_is_fallback": [False, False, False, False, False, False],
        "svm_regime_is_no_trade": [False, False, False, False, False, False],
        "svm_regime_is_empty": [False, False, False, False, False, False],
    }


def _evidence(path: Path, *, parity: float = 1.0) -> Path:
    terminal_frame = _terminal_frame()
    artifact = path.parent / "svm-regime-handoff.joblib"
    artifact.write_bytes(b"trusted artifact placeholder")
    path.write_text(
        json.dumps(
            {
                "handoff_classifier_path": str(artifact),
                "handoff_classifier_sha256": _sha256_file(artifact),
                "handoff_parity_match_rate": parity,
                "test_rows": 6,
                "trained_terminal_names": list(REQUIRED_TRAINED_TERMINALS),
                "trained_terminal_frame": terminal_frame,
                "trained_terminal_counts": {
                    key: sum(values) for key, values in terminal_frame.items()
                },
            }
        )
    )
    return path


def test_run_paired_detector_swap_writes_evolved_trained_and_comparison(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path / "detector_swap.json")
    result = run_paired_detector_swap(
        seed=7,
        evidence_path=evidence,
        output_root=tmp_path / "out",
        population_size=10,
        generations=2,
        trusted_classifier_path=tmp_path / "svm-regime-handoff.joblib",
    )

    assert result["seed"] == 7
    assert result["trusted_classifier_path"] == str(
        tmp_path / "svm-regime-handoff.joblib"
    )
    assert result["trained_terminal_names"] == list(REQUIRED_TRAINED_TERMINALS)
    assert (tmp_path / "out" / "evolved" / "seed-7" / "evolution_summary.json").exists()
    assert (tmp_path / "out" / "trained" / "seed-7" / "evolution_summary.json").exists()
    comparison = tmp_path / "out" / "comparison" / "seed-7" / "paired_evolution.json"
    assert comparison.exists()
    saved = json.loads(comparison.read_text())
    assert saved["evolved"]["detector_source"] == "evolved"
    assert saved["trained"]["detector_source"] == "trained"
    assert len(saved["trained"]["fitness_curve"]) == 2
    assert saved["trained"]["time_to_final_best_generation"] is not None
    assert "final_mean_fitness_delta_trained_minus_evolved" in saved


def test_run_paired_detector_swap_rejects_counts_only_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "bad.json"
    evidence.write_text(
        json.dumps({"trained_terminal_counts": {"svm_regime_is_trend": 2}})
    )

    with pytest.raises(ValueError, match="trained_terminal_frame"):
        run_paired_detector_swap(
            seed=7, evidence_path=evidence, output_root=tmp_path / "out"
        )


def test_run_paired_detector_swap_enforces_handoff_parity_gate(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path / "detector_swap.json", parity=0.99)

    with pytest.raises(ValueError, match="handoff parity"):
        run_paired_detector_swap(
            seed=7, evidence_path=evidence, output_root=tmp_path / "out"
        )


def test_run_paired_detector_swap_requires_all_six_trained_terminals(
    tmp_path: Path,
) -> None:
    terminal_frame = _terminal_frame()
    terminal_frame.pop("svm_regime_is_empty")
    evidence = tmp_path / "detector_swap.json"
    evidence.write_text(
        json.dumps(
            {
                "handoff_parity_match_rate": 1.0,
                "trained_terminal_names": list(terminal_frame),
                "trained_terminal_frame": terminal_frame,
            }
        )
    )

    with pytest.raises(ValueError, match="missing trained terminal"):
        run_paired_detector_swap(
            seed=7, evidence_path=evidence, output_root=tmp_path / "out"
        )


def test_run_paired_detector_swap_rejects_non_list_terminal_names(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "detector_swap.json"
    evidence.write_text(
        json.dumps(
            {
                "handoff_parity_match_rate": 1.0,
                "trained_terminal_names": "svm_regime_is_trend",
                "trained_terminal_frame": _terminal_frame(),
            }
        )
    )

    with pytest.raises(ValueError, match="trained_terminal_names"):
        run_paired_detector_swap(
            seed=7, evidence_path=evidence, output_root=tmp_path / "out"
        )


def test_run_paired_detector_swap_requires_canonical_terminal_order(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "detector_swap.json"
    evidence.write_text(
        json.dumps(
            {
                "handoff_parity_match_rate": 1.0,
                "trained_terminal_names": list(reversed(REQUIRED_TRAINED_TERMINALS)),
                "trained_terminal_frame": _terminal_frame(),
            }
        )
    )

    with pytest.raises(ValueError, match="canonical order"):
        run_paired_detector_swap(
            seed=7, evidence_path=evidence, output_root=tmp_path / "out"
        )


def test_run_paired_detector_swap_rejects_non_sequence_terminal_values(
    tmp_path: Path,
) -> None:
    terminal_frame: dict[str, object] = dict(_terminal_frame())
    terminal_frame["svm_regime_is_trend"] = "not-a-sequence"
    evidence = tmp_path / "detector_swap.json"
    evidence.write_text(
        json.dumps(
            {
                "handoff_parity_match_rate": 1.0,
                "trained_terminal_names": list(REQUIRED_TRAINED_TERMINALS),
                "trained_terminal_frame": terminal_frame,
            }
        )
    )

    with pytest.raises(ValueError, match="must be a sequence"):
        run_paired_detector_swap(
            seed=7, evidence_path=evidence, output_root=tmp_path / "out"
        )


def test_run_paired_detector_swap_rejects_empty_terminal_series(tmp_path: Path) -> None:
    terminal_frame = _terminal_frame()
    terminal_frame["svm_regime_is_trend"] = []
    evidence = tmp_path / "detector_swap.json"
    evidence.write_text(
        json.dumps(
            {
                "handoff_parity_match_rate": 1.0,
                "trained_terminal_names": list(REQUIRED_TRAINED_TERMINALS),
                "trained_terminal_frame": terminal_frame,
            }
        )
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        run_paired_detector_swap(
            seed=7, evidence_path=evidence, output_root=tmp_path / "out"
        )


def test_run_paired_detector_swap_rejects_mismatched_terminal_lengths(
    tmp_path: Path,
) -> None:
    terminal_frame = _terminal_frame()
    terminal_frame["svm_regime_is_range"] = [False]
    evidence = tmp_path / "detector_swap.json"
    evidence.write_text(
        json.dumps(
            {
                "handoff_parity_match_rate": 1.0,
                "trained_terminal_names": list(REQUIRED_TRAINED_TERMINALS),
                "trained_terminal_frame": terminal_frame,
            }
        )
    )

    with pytest.raises(ValueError, match="same length"):
        run_paired_detector_swap(
            seed=7, evidence_path=evidence, output_root=tmp_path / "out"
        )


def test_run_paired_detector_swap_rejects_non_finite_parity(tmp_path: Path) -> None:
    evidence = tmp_path / "detector_swap.json"
    evidence.write_text(
        json.dumps(
            {
                "handoff_parity_match_rate": "1.0",
                "trained_terminal_names": list(REQUIRED_TRAINED_TERMINALS),
                "trained_terminal_frame": _terminal_frame(),
            }
        )
    )

    with pytest.raises(ValueError, match="finite handoff_parity_match_rate"):
        run_paired_detector_swap(
            seed=7, evidence_path=evidence, output_root=tmp_path / "out"
        )


def test_run_paired_detector_swap_rejects_invalid_handoff_metadata(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "detector_swap.json"
    evidence.write_text(
        json.dumps(
            {
                "handoff_classifier_path": ["not", "a", "string"],
                "handoff_parity_match_rate": 1.0,
                "trained_terminal_names": list(REQUIRED_TRAINED_TERMINALS),
                "trained_terminal_frame": _terminal_frame(),
            }
        )
    )

    with pytest.raises(ValueError, match="handoff_classifier_path"):
        run_paired_detector_swap(
            seed=7, evidence_path=evidence, output_root=tmp_path / "out"
        )


def test_run_paired_detector_swap_rejects_invalid_sha_metadata(tmp_path: Path) -> None:
    evidence = tmp_path / "detector_swap.json"
    evidence.write_text(
        json.dumps(
            {
                "handoff_classifier_sha256": ["not", "a", "string"],
                "handoff_parity_match_rate": 1.0,
                "trained_terminal_names": list(REQUIRED_TRAINED_TERMINALS),
                "trained_terminal_frame": _terminal_frame(),
            }
        )
    )

    with pytest.raises(ValueError, match="handoff_classifier_sha256"):
        run_paired_detector_swap(
            seed=7, evidence_path=evidence, output_root=tmp_path / "out"
        )


def test_run_paired_detector_swap_requires_existing_trusted_artifact(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path / "detector_swap.json")

    with pytest.raises(ValueError, match="does not exist"):
        run_paired_detector_swap(
            seed=7,
            evidence_path=evidence,
            output_root=tmp_path / "out",
            trusted_classifier_path=tmp_path / "missing.joblib",
        )


def test_run_paired_detector_swap_requires_sha_when_trusted_artifact_provided(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "svm-regime-handoff.joblib"
    artifact.write_bytes(b"trusted artifact placeholder")
    evidence = tmp_path / "detector_swap.json"
    evidence.write_text(
        json.dumps(
            {
                "handoff_classifier_path": str(artifact),
                "handoff_parity_match_rate": 1.0,
                "trained_terminal_names": list(REQUIRED_TRAINED_TERMINALS),
                "trained_terminal_frame": _terminal_frame(),
            }
        )
    )

    with pytest.raises(ValueError, match="missing handoff_classifier_sha256"):
        run_paired_detector_swap(
            seed=7,
            evidence_path=evidence,
            output_root=tmp_path / "out",
            trusted_classifier_path=artifact,
        )


def test_run_paired_detector_swap_rejects_trusted_artifact_hash_mismatch(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path / "detector_swap.json")
    artifact = tmp_path / "svm-regime-handoff.joblib"
    artifact.write_bytes(b"mutated artifact")

    with pytest.raises(ValueError, match="hash does not match"):
        run_paired_detector_swap(
            seed=7,
            evidence_path=evidence,
            output_root=tmp_path / "out",
            trusted_classifier_path=artifact,
        )
