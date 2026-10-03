#!/usr/bin/env python3
"""Validate key Round-2 statistical claims against archived JSON artifacts."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def _report(path: Path, threshold: float) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return next(report for report in payload["reports"] if float(report["iou_threshold"]) == threshold)


def _assert_close(actual: float, expected: float, label: str, tolerance: float = 5e-5) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def validate(root: Path) -> dict:
    reports = root / "data" / "reports"
    strict_summary = json.loads(
        (reports / "round2_jepa_graph_strict_nested_compact_cv5.summary.json").read_text(encoding="utf-8")
    )
    strict_03 = _report(reports / "round2_jepa_graph_strict_nested_compact_expanded.json", 0.3)
    strict_05 = _report(reports / "round2_jepa_graph_strict_nested_compact_expanded.json", 0.5)
    matched_summary = json.loads(
        (reports / "round2_jepa_actionformer_matched_cv5.summary.json").read_text(encoding="utf-8")
    )
    matched_03 = _report(reports / "round2_jepa_actionformer_matched_expanded.json", 0.3)
    matched_05 = _report(reports / "round2_jepa_actionformer_matched_expanded.json", 0.5)
    matched_vs_videomae_05 = json.loads(
        (reports / "round2_jepa_actionformer_vs_videomae_actionformer_bootstrap_iou05.json").read_text(
            encoding="utf-8"
        )
    )
    occupancy_summary = json.loads(
        (reports / "round2_jepa_occupancy_actionformer_cv5.summary.json").read_text(encoding="utf-8")
    )
    occupancy_03 = _report(reports / "round2_jepa_occupancy_actionformer_expanded.json", 0.3)
    occupancy_vs_full_span_03 = json.loads(
        (reports / "round2_jepa_occupancy_vs_full_span_bootstrap_iou03.json").read_text(encoding="utf-8")
    )
    official_actionformer_03 = _report(reports / "round2_official_actionformer_r3d_expanded.json", 0.3)
    official_actionformer_05 = _report(reports / "round2_official_actionformer_r3d_expanded.json", 0.5)
    jepa_vs_official_03 = json.loads(
        (reports / "round2_jepa_matched_vs_official_actionformer_r3d_bootstrap_iou03.json").read_text(encoding="utf-8")
    )
    jepa_vs_official_05 = json.loads(
        (reports / "round2_jepa_matched_vs_official_actionformer_r3d_bootstrap_iou05.json").read_text(encoding="utf-8")
    )
    official_calibration = json.loads(
        (reports / "round2_official_actionformer_r3d_cv5.calibration.json").read_text(encoding="utf-8")
    )
    official_zscore_jepa_03 = _report(reports / "round2_official_actionformer_jepa_zscore_expanded.json", 0.3)
    official_zscore_jepa_05 = _report(reports / "round2_official_actionformer_jepa_zscore_expanded.json", 0.5)
    official_zscore_videomae_03 = _report(reports / "round2_official_actionformer_videomae_zscore_expanded.json", 0.3)
    official_zscore_r3d_03 = _report(reports / "round2_official_actionformer_r3d_zscore_expanded.json", 0.3)
    zscore_jepa_vs_videomae_03 = json.loads(
        (reports / "round2_official_actionformer_zscore_jepa_vs_videomae_iou03.json").read_text(encoding="utf-8")
    )
    zscore_jepa_vs_videomae_05 = json.loads(
        (reports / "round2_official_actionformer_zscore_jepa_vs_videomae_iou05.json").read_text(encoding="utf-8")
    )
    zscore_jepa_vs_r3d_03 = json.loads(
        (reports / "round2_official_actionformer_zscore_jepa_vs_r3d_iou03.json").read_text(encoding="utf-8")
    )

    checks = {
        "strict_fold_mean_f1_iou03": (strict_summary["aggregate"]["segment"]["f1_mean"], 0.7541062834427248),
        "strict_pooled_f1_iou03": (strict_03["overall"]["segment"]["f1"], 0.7480309965360727),
        "strict_pooled_f1_iou05": (strict_05["overall"]["segment"]["f1"], 0.6220467450636366),
        "matched_fold_mean_f1_iou03": (matched_summary["aggregate"]["segment"]["f1_mean"], 0.7732333997805741),
        "matched_pooled_f1_iou03": (matched_03["overall"]["segment"]["f1"], 0.7704585835438102),
        "matched_pooled_f1_iou05": (matched_05["overall"]["segment"]["f1"], 0.6546901209321636),
        "matched_gain_over_videomae_iou05": (matched_vs_videomae_05["observed_difference"], 0.03256117621832366),
        "matched_gain_ci_low_iou05": (matched_vs_videomae_05["difference_ci95"][0], 0.0017850789714514703),
        "matched_gain_ci_high_iou05": (matched_vs_videomae_05["difference_ci95"][1], 0.06494031189220428),
        "occupancy_fold_mean_f1_iou03": (occupancy_summary["aggregate"]["segment"]["f1_mean"], 0.7665201122578422),
        "occupancy_pooled_f1_iou03": (occupancy_03["overall"]["segment"]["f1"], 0.7636358667241231),
        "occupancy_difference_from_full_span_iou03": (
            occupancy_vs_full_span_03["observed_difference"],
            -0.007792209052950971,
        ),
        "official_actionformer_pooled_f1_iou03": (
            official_actionformer_03["overall"]["segment"]["f1"],
            0.773005639588639,
        ),
        "official_actionformer_pooled_f1_iou05": (
            official_actionformer_05["overall"]["segment"]["f1"],
            0.6503062535874686,
        ),
        "matched_jepa_difference_from_official_actionformer_iou03": (
            jepa_vs_official_03["observed_difference"],
            -0.002547056044828744,
        ),
        "matched_jepa_difference_from_official_actionformer_iou05": (
            jepa_vs_official_05["observed_difference"],
            0.0043838673446949805,
        ),
        "official_zscore_jepa_pooled_f1_iou03": (official_zscore_jepa_03["overall"]["segment"]["f1"], 0.7663929475530569),
        "official_zscore_jepa_pooled_f1_iou05": (official_zscore_jepa_05["overall"]["segment"]["f1"], 0.6393437677459443),
        "official_zscore_videomae_pooled_f1_iou03": (official_zscore_videomae_03["overall"]["segment"]["f1"], 0.74590114435836),
        "official_zscore_r3d_pooled_f1_iou03": (official_zscore_r3d_03["overall"]["segment"]["f1"], 0.7673464431406744),
        "official_zscore_jepa_difference_from_videomae_iou03": (zscore_jepa_vs_videomae_03["observed_difference"], 0.0204918031946969),
    }
    for label, (actual, expected) in checks.items():
        _assert_close(actual, expected, label)

    if any(
        fold[stage]["config"].get("enabled", False)
        for fold in strict_summary["folds"]
        for stage in ("proposal_replacement", "graph_reranker", "protected_fusion")
    ):
        raise AssertionError("strict graph artifact unexpectedly enables a rejected postprocessing stage")
    if any(not math.isfinite(fold["training"]["best_calibration_objective"]) for fold in occupancy_summary["folds"]):
        raise AssertionError("occupancy-aware artifact contains a non-finite calibration objective")
    if len(official_calibration["folds"]) != 5 or any(
        fold["params"]["max_predictions"] != 1 for fold in official_calibration["folds"]
    ):
        raise AssertionError("official ActionFormer calibration audit is incomplete or unexpected")
    if not (jepa_vs_official_03["difference_ci95"][0] < 0 < jepa_vs_official_03["difference_ci95"][1]):
        raise AssertionError("IoU-0.3 JEPA/official-ActionFormer CI no longer supports a statistical tie")
    if not (jepa_vs_official_05["difference_ci95"][0] < 0 < jepa_vs_official_05["difference_ci95"][1]):
        raise AssertionError("IoU-0.5 JEPA/official-ActionFormer CI no longer supports a statistical tie")
    for label, comparison in (
        ("JEPA/VideoMAE IoU-0.3", zscore_jepa_vs_videomae_03),
        ("JEPA/VideoMAE IoU-0.5", zscore_jepa_vs_videomae_05),
        ("JEPA/R3D IoU-0.3", zscore_jepa_vs_r3d_03),
    ):
        if not (comparison["difference_ci95"][0] < 0 < comparison["difference_ci95"][1]):
            raise AssertionError(f"official normalized {label} comparison no longer supports a statistical tie")

    document = (root / "docs" / "round2_statistical_evidence_report.md").read_text(encoding="utf-8")
    required_snippets = [
        "0.7541",
        "0.7732",
        "+0.0326",
        "[0.0018, 0.0649]",
        "outer-fold-assisted development",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in document]
    if missing:
        raise AssertionError(f"statistical report is missing expected snippets: {missing}")
    occupancy_document = (root / "docs" / "round2_occupancy_aware_experiment.md").read_text(encoding="utf-8")
    occupancy_snippets = ["0.7636", "[-0.0152, -0.0016]", "181/215", "Reject occupancy/event-count"]
    missing_occupancy = [snippet for snippet in occupancy_snippets if snippet not in occupancy_document]
    if missing_occupancy:
        raise AssertionError(f"occupancy report is missing expected snippets: {missing_occupancy}")
    official_document = (root / "docs" / "round2_external_baseline_report.md").read_text(encoding="utf-8")
    official_snippets = ["0.7664", "0.7459", "statistically tied", "head-dependent"]
    missing_official = [snippet for snippet in official_snippets if snippet not in official_document]
    if missing_official:
        raise AssertionError(f"official matched-representation report is missing expected snippets: {missing_official}")
    active_documents = [
        "round2_rebuttal_draft.md",
        "round2_statistical_evidence_report.md",
        "round2_external_baseline_report.md",
        "reviewer_progress_and_next_experiment.md",
        "round2_revision_checklist.md",
        "next_experiment_videomae_protocol.md",
        "round2_protocol.md",
        "JEPA_error_segment_paper_draft_en.md",
        "JEPA_error_segment_paper_draft_zh.md",
    ]
    invalid_claims = []
    historical_mentions = 0
    for name in active_documents:
        for line_number, line in enumerate((root / "docs" / name).read_text(encoding="utf-8").splitlines(), 1):
            if "0.8208" not in line:
                continue
            historical_mentions += 1
            lower = line.lower()
            calls_strict = bool(re.search(r"strict|nested|严格", lower))
            explicitly_disclaimed = bool(
                re.search(r"historical|outer-fold|not strict|never strict|不是严格|历史", lower)
            )
            if calls_strict and not explicitly_disclaimed:
                invalid_claims.append(f"{name}:{line_number}: {line.strip()}")
    if invalid_claims:
        raise AssertionError("0.8208 is still described as strict/nested:\n" + "\n".join(invalid_claims))

    return {
        "checks": len(checks),
        "document_snippets": len(required_snippets),
        "occupancy_document_snippets": len(occupancy_snippets),
        "official_document_snippets": len(official_snippets),
        "historical_0.8208_mentions_reviewed": historical_mentions,
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
