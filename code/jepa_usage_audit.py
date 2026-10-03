#!/usr/bin/env python3
"""
Audit whether the current pipeline uses JEPA objectives or encoder proxies.

This is intentionally static and lightweight: it does not load checkpoints or
run CUDA. Its job is to keep experimental claims honest before expensive runs.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditFinding:
    component: str
    status: str
    detail: str


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise ValueError(f"Function {name!r} not found")


def _has_predictor_forward(source: str) -> bool:
    compact = source.replace(" ", "")
    return any(
        pattern in compact
        for pattern in (
            "predictor(",
            ".predictor(",
            "predictor.forward(",
        )
    )


def audit_detect_pipeline(path: Path) -> list[AuditFinding]:
    source = _read_source(path)
    findings: list[AuditFinding] = []

    vjepa_extract = _function_source(source, "extract_vjepa_tokens")
    vjepa_score = _function_source(source, "score_vjepa_multi")
    ijepa_extract = _function_source(source, "extract_ijepa_all")
    ijepa_score = _function_source(source, "score_ijepa_multi")

    uses_vjepa_encoder = "encoder(batch)" in vjepa_extract
    uses_vjepa_predictor_forward = _has_predictor_forward(vjepa_extract) or _has_predictor_forward(vjepa_score)
    uses_masks = "mask" in vjepa_extract.lower() or "mask" in vjepa_score.lower()
    uses_cosine_proxy = "cosine_similarity" in vjepa_score or "np.dot" in vjepa_score

    if uses_vjepa_encoder and uses_cosine_proxy and not uses_vjepa_predictor_forward:
        findings.append(
            AuditFinding(
                "V-JEPA",
                "proxy",
                "Uses encoder token distances; no V-JEPA predictor forward pass was found.",
            )
        )
    elif uses_vjepa_predictor_forward and uses_masks:
        findings.append(
            AuditFinding(
                "V-JEPA",
                "objective",
                "Predictor and masks are present; this looks like JEPA prediction scoring.",
            )
        )
    else:
        findings.append(
            AuditFinding(
                "V-JEPA",
                "unclear",
                "Could not prove whether predictor-based JEPA scoring is used.",
            )
        )

    constructs_ijepa_encoder = "PatchEmbed" in ijepa_extract and "for blk in i_model.blocks" in ijepa_extract
    uses_ijepa_predictor = "vit_predictor" in ijepa_extract or _has_predictor_forward(ijepa_extract) or _has_predictor_forward(ijepa_score)
    uses_patch_stats = all(token in ijepa_score for token in ["var(dim=1)", "spatial_inc", "energy"])

    if constructs_ijepa_encoder and uses_patch_stats and not uses_ijepa_predictor:
        findings.append(
            AuditFinding(
                "I-JEPA",
                "proxy",
                "Uses encoder patch statistics; no I-JEPA predictor/reconstruction objective was found.",
            )
        )
    elif uses_ijepa_predictor:
        findings.append(
            AuditFinding(
                "I-JEPA",
                "objective",
                "Predictor usage is present; inspect masks and target comparison manually.",
            )
        )
    else:
        findings.append(
            AuditFinding(
                "I-JEPA",
                "unclear",
                "Could not prove whether predictor-based I-JEPA scoring is used.",
            )
        )

    if "norms_block[i]" in vjepa_extract:
        findings.append(
            AuditFinding(
                "V-JEPA hierarchical",
                "suspicious",
                "Hooks encoder.norms_block[i] for i=0..3, not documented layers [4, 11, 17, 23].",
            )
        )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline",
        default=str(Path(__file__).with_name("detect_and_report_v4.py")),
        help="Path to detect_and_report_v4.py",
    )
    args = parser.parse_args()

    findings = audit_detect_pipeline(Path(args.pipeline))
    for finding in findings:
        print(f"{finding.component}: {finding.status} - {finding.detail}")

    if any(f.status in {"proxy", "suspicious", "unclear"} for f in findings):
        print("\nConclusion: do not claim JEPA objective/prediction use without fixing or qualifying this.")
        return 2
    print("\nConclusion: JEPA objective usage is present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
