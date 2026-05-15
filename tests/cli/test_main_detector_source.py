"""Tests for trained detector CLI flag validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from liq.evolution.cli.main import main
from liq.evolution.phase4_detector_swap import REQUIRED_TRAINED_TERMINALS


def _write_phase4_evidence(tmp_path: Path) -> tuple[Path, Path]:
    artifact = tmp_path / "svm-regime-handoff.joblib"
    artifact.write_bytes(b"trusted artifact placeholder")
    terminal_frame = {
        "svm_regime_is_trend": [True, False, False, True, False, False],
        "svm_regime_is_range": [False, True, False, False, True, False],
        "svm_regime_is_neutral": [False, False, True, False, False, True],
        "svm_regime_is_fallback": [False, False, False, False, False, False],
        "svm_regime_is_no_trade": [False, False, False, False, False, False],
        "svm_regime_is_empty": [False, False, False, False, False, False],
    }
    evidence = tmp_path / "detector_swap.json"
    evidence.write_text(
        json.dumps(
            {
                "handoff_classifier_path": str(artifact),
                "handoff_classifier_sha256": hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest(),
                "handoff_parity_match_rate": 1.0,
                "trained_terminal_names": list(REQUIRED_TRAINED_TERMINALS),
                "trained_terminal_frame": terminal_frame,
            }
        )
    )
    return evidence, artifact


def test_detector_source_evolved_is_default(capsys) -> None:
    assert (
        main(["--population-size", "10", "--generations", "1", "--max-depth", "2"]) == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True


def test_detector_source_trained_requires_trusted_artifact_path(capsys) -> None:
    code = main(
        [
            "--detector-source",
            "trained",
            "--population-size",
            "10",
            "--generations",
            "1",
            "--max-depth",
            "2",
        ]
    )

    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error_code"] == "evo.cli.trained_detector_missing_artifact"
    assert "trusted" in error["remediation"]


def test_detector_source_trained_accepts_handoff_path(tmp_path: Path, capsys) -> None:
    artifact = tmp_path / "svm-regime-handoff.joblib"
    artifact.write_bytes(b"trusted artifact placeholder")

    code = main(
        [
            "--detector-source",
            "trained",
            "--handoff-classifier-path",
            str(artifact),
            "--population-size",
            "10",
            "--generations",
            "1",
            "--max-depth",
            "2",
        ]
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True


def test_phase4_selected_source_requires_both_paths(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "--phase4-evidence-path",
            str(tmp_path / "detector_swap.json"),
            "--population-size",
            "10",
            "--generations",
            "1",
            "--max-depth",
            "2",
        ]
    )

    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error_code"] == "evo.cli.phase4_paths_missing"


def test_phase4_selected_source_runs_only_requested_trained_branch(
    tmp_path: Path, capsys
) -> None:
    evidence, artifact = _write_phase4_evidence(tmp_path)
    output_dir = tmp_path / "out"

    code = main(
        [
            "--detector-source",
            "trained",
            "--handoff-classifier-path",
            str(artifact),
            "--phase4-evidence-path",
            str(evidence),
            "--phase4-output-dir",
            str(output_dir),
            "--population-size",
            "10",
            "--generations",
            "1",
            "--max-depth",
            "2",
        ]
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    selected = output["phase4_detector_source_evolution"]
    assert selected["detector_source"] == "trained"
    assert selected["trained_terminal_names"] == list(REQUIRED_TRAINED_TERMINALS)
    assert (output_dir / "trained" / "seed-42" / "evolution_summary.json").exists()
    assert not (output_dir / "evolved" / "seed-42" / "evolution_summary.json").exists()
    assert output["phase4_paired_evolution"] is None


def test_phase4_selected_source_runs_only_requested_evolved_branch(
    tmp_path: Path, capsys
) -> None:
    evidence, _artifact = _write_phase4_evidence(tmp_path)
    output_dir = tmp_path / "out"

    code = main(
        [
            "--detector-source",
            "evolved",
            "--phase4-evidence-path",
            str(evidence),
            "--phase4-output-dir",
            str(output_dir),
            "--population-size",
            "10",
            "--generations",
            "1",
            "--max-depth",
            "2",
        ]
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    selected = output["phase4_detector_source_evolution"]
    assert selected["detector_source"] == "evolved"
    assert (output_dir / "evolved" / "seed-42" / "evolution_summary.json").exists()
    assert not (output_dir / "trained" / "seed-42" / "evolution_summary.json").exists()
    assert output["phase4_paired_evolution"] is None


def test_phase4_paired_comparison_requires_trusted_artifact_path(
    tmp_path: Path, capsys
) -> None:
    evidence, _artifact = _write_phase4_evidence(tmp_path)

    code = main(
        [
            "--phase4-evidence-path",
            str(evidence),
            "--phase4-output-dir",
            str(tmp_path / "out"),
            "--phase4-paired-comparison",
            "--population-size",
            "10",
            "--generations",
            "1",
            "--max-depth",
            "2",
        ]
    )

    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error_code"] == "evo.cli.phase4_trusted_artifact_missing"
    assert "SHA-256" in error["remediation"]


def test_phase4_paired_comparison_is_explicit(tmp_path: Path, capsys) -> None:
    evidence, artifact = _write_phase4_evidence(tmp_path)
    output_dir = tmp_path / "out"

    code = main(
        [
            "--detector-source",
            "trained",
            "--handoff-classifier-path",
            str(artifact),
            "--phase4-evidence-path",
            str(evidence),
            "--phase4-output-dir",
            str(output_dir),
            "--phase4-paired-comparison",
            "--population-size",
            "10",
            "--generations",
            "1",
            "--max-depth",
            "2",
        ]
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["phase4_detector_source_evolution"] is None
    assert output["phase4_paired_evolution"]["trusted_classifier_path"] == str(artifact)
    assert (output_dir / "comparison" / "seed-42" / "paired_evolution.json").exists()
