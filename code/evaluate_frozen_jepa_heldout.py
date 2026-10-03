#!/usr/bin/env python3
"""Evaluate the frozen JEPA-Loc pipeline on a held-out dataset.

The evaluator intentionally does not select thresholds or tune fusion on the
held-out set. It applies checkpoint/config artifacts frozen on full train.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

import train_proposal_set_selector as proposal_selector_module
from jepa_graph_reranker import select_graph_reranked_predictions
from jepa_prediction_set_switcher import apply_set_switch_config
from selector_fusion import evaluate_fused_predictions, fuse_protected_segment_predictions
from train_proposal_set_selector import (
    _outside_values,
    _topk_mean,
    extract_set_proposal_features,
    generate_multichannel_candidates,
    select_weighted_proposal_set,
)
from train_segment_locator import (
    TemporalSegmentLocator,
    _segments_from_params,
    blend_prediction_records,
    evaluate_records,
    load_signal_dataset,
    predict_records,
)


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "__main__" and hasattr(proposal_selector_module, name):
            return getattr(proposal_selector_module, name)
        return super().find_class(module, name)


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_feature_names(data_dir: str | Path) -> list[str]:
    summary = _read_json(Path(data_dir) / "summary.json")
    return [str(name) for name in summary.get("feature_names", [])]


def _load_model(checkpoint_path: str | Path, device: str) -> tuple[TemporalSegmentLocator, np.ndarray, np.ndarray, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    model = TemporalSegmentLocator(
        in_channels=int(config.get("in_channels", len(checkpoint["normalizer"]["mean"]))),
        hidden=int(config.get("hidden", 32)),
        dropout=float(config.get("dropout", 0.1)),
        architecture=str(config.get("architecture", "tcn")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    normalizer = checkpoint["normalizer"]
    return (
        model,
        np.asarray(normalizer["mean"], dtype=np.float32),
        np.asarray(normalizer["std"], dtype=np.float32),
        checkpoint,
    )


def _legacy_94_set_proposal_features(
    signals: np.ndarray,
    segment: tuple[int, int],
    channel_indices: list[int],
) -> np.ndarray:
    """Reproduce the 94-dim proposal features used by frozen full-train pickles."""
    signals = np.asarray(signals, dtype=np.float32)
    n = len(signals)
    start, end = segment
    start = max(0, min(int(start), n - 1))
    end = max(start, min(int(end), n - 1))
    duration = end - start + 1
    features: list[float] = [
        float(duration),
        float(np.log1p(duration)),
        float(start / max(1, n - 1)),
        float(end / max(1, n - 1)),
        float((start + end) / max(1, 2 * (n - 1))),
    ]
    per_channel_means: list[float] = []
    per_channel_maxes: list[float] = []
    for idx in channel_indices:
        values = np.nan_to_num(signals[:, int(idx)], nan=0.0, posinf=0.0, neginf=0.0)
        inside = values[start : end + 1]
        outside = _outside_values(values, start, end)
        contrast = float(inside.mean() - outside.mean()) if len(outside) else 0.0
        slope = float(inside[-1] - inside[0]) / max(1, duration - 1)
        per_channel_means.append(float(inside.mean()))
        per_channel_maxes.append(float(inside.max()))
        features.extend(
            [
                float(inside.mean()),
                float(inside.max()),
                _topk_mean(inside),
                float(inside.std()),
                float(inside.max() - inside.min()),
                slope,
                contrast,
            ]
        )
    if per_channel_means:
        features.extend(
            [
                float(np.mean(per_channel_means)),
                float(np.max(per_channel_means)),
                float(np.std(per_channel_means)),
                float(np.mean(per_channel_maxes)),
                float(np.max(per_channel_maxes)),
            ]
        )
    return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _score_video_candidates_frozen(
    classifier,
    record,
    feature_names: list[str],
    channel_names: list[str],
    thresholds: list[float],
    min_gaps: list[int],
    min_lengths: list[int],
) -> tuple[list[tuple[int, int]], np.ndarray, str]:
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    channel_indices = [name_to_idx[name] for name in channel_names if name in name_to_idx]
    candidates = generate_multichannel_candidates(
        record.signals,
        feature_names=feature_names,
        channel_names=channel_names,
        thresholds=thresholds,
        min_gaps=min_gaps,
        min_lengths=min_lengths,
    )
    if not candidates:
        return [], np.zeros(0, dtype=np.float32), "empty"
    expected_dim = int(
        getattr(classifier, "n_features_in_", 0)
        or getattr(getattr(classifier, "model", None), "n_features_in_", 0)
        or 0
    )
    if expected_dim == 94:
        features = [_legacy_94_set_proposal_features(record.signals, segment, channel_indices) for segment in candidates]
        mode = "legacy_94"
    else:
        features = [extract_set_proposal_features(record.signals, segment, channel_indices) for segment in candidates]
        mode = "current"
    x = np.stack(features).astype(np.float32)
    if expected_dim and x.shape[1] != expected_dim:
        raise ValueError(f"candidate feature dim {x.shape[1]} does not match frozen classifier dim {expected_dim}")
    probabilities = classifier.predict_proba(x)[:, 1].astype(np.float32)
    return candidates, probabilities, mode


def _mainline_outputs(
    model: TemporalSegmentLocator,
    records,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    checkpoint_metrics: dict,
    feature_names: list[str],
) -> tuple[list[dict], list[list[tuple[int, int]]]]:
    outputs = predict_records(model, records, range(len(records)), mean, std, device)
    hybrid = checkpoint_metrics.get("hybrid", {})
    if hybrid.get("aux_name"):
        aux_name = str(hybrid["aux_name"])
        if aux_name not in feature_names:
            raise ValueError(f"hybrid aux feature {aux_name!r} not found in held-out feature names")
        outputs = blend_prediction_records(
            outputs,
            records,
            range(len(records)),
            aux_channel=feature_names.index(aux_name),
            alpha_model=float(hybrid.get("alpha_model", 1.0)),
        )
    params = checkpoint_metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1})
    predictions = [_segments_from_params(np.asarray(item["probs"], dtype=np.float32), params) for item in outputs]
    return outputs, predictions


def _proposal_predictions(
    selector_path: str | Path,
    records,
    feature_names: list[str],
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]], list[np.ndarray], dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.modules.setdefault("train_proposal_set_selector", proposal_selector_module)
    with Path(selector_path).open("rb") as handle:
        selector = _CompatUnpickler(handle).load()
    classifier = selector["classifier"]
    channel_names = [str(name) for name in selector["channel_names"] if str(name) in feature_names]
    thresholds = [float(value) for value in selector["thresholds"]]
    min_gaps = [int(value) for value in selector["min_gaps"]]
    min_lengths = [int(value) for value in selector["min_lengths"]]
    params = selector["params"]
    predictions: list[list[tuple[int, int]]] = []
    candidates_by_video: list[list[tuple[int, int]]] = []
    scores_by_video: list[np.ndarray] = []
    feature_modes: set[str] = set()
    for record in records:
        candidates, scores, feature_mode = _score_video_candidates_frozen(
            classifier,
            record,
            feature_names=feature_names,
            channel_names=channel_names,
            thresholds=thresholds,
            min_gaps=min_gaps,
            min_lengths=min_lengths,
        )
        feature_modes.add(feature_mode)
        candidates_by_video.append(candidates)
        scores_by_video.append(scores.astype(np.float32))
        selected = []
        selected_scores = []
        for segment, probability in zip(candidates, scores):
            if float(probability) >= float(params["prob_threshold"]):
                selected.append(segment)
                selected_scores.append(float(probability))
        predictions.append(
            select_weighted_proposal_set(
                selected,
                selected_scores,
                length_penalty=float(params["length_penalty"]),
            )
        )
    export = {
        "selector_path": str(selector_path),
        "channel_names": channel_names,
        "thresholds": thresholds,
        "min_gaps": min_gaps,
        "min_lengths": min_lengths,
        "params": params,
        "candidate_counts": [len(items) for items in candidates_by_video],
        "feature_modes": sorted(feature_modes),
        "classifier_n_features_in": int(getattr(classifier, "n_features_in_", 0) or 0),
    }
    return predictions, candidates_by_video, scores_by_video, export


def _graph_predictions(
    mainline_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    graph_config: dict,
) -> list[list[tuple[int, int]]]:
    if not graph_config.get("enabled", False):
        return [[] for _ in mainline_predictions]
    return select_graph_reranked_predictions(
        mainline_predictions,
        candidate_predictions,
        candidate_scores,
        prob_threshold=float(graph_config["prob_threshold"]),
        support_iou=float(graph_config["support_iou"]),
        support_weight=float(graph_config["support_weight"]),
        support_count_weight=float(graph_config["support_count_weight"]),
        split_penalty=float(graph_config["split_penalty"]),
        mainline_overlap_penalty=float(graph_config["mainline_overlap_penalty"]),
        length_penalty=float(graph_config["length_penalty"]),
    )


def _select_frozen_graph_config(graph_summary: dict) -> tuple[dict, dict]:
    """Select one deployment graph config from already-frozen train-CV artifacts."""
    candidates: list[tuple[tuple[float, float, float, float], dict, dict]] = []
    for fold in graph_summary.get("folds", []):
        graph = fold.get("fusion", {}).get("graph_reranker", {})
        config = graph.get("config", graph) if isinstance(graph, dict) else {}
        if not isinstance(config, dict) or not config.get("enabled", False):
            continue
        segment = graph.get("metrics", {}).get("segment", {}) if isinstance(graph, dict) else {}
        key = (
            float(segment.get("f1", 0.0)),
            float(segment.get("precision", 0.0)),
            float(segment.get("recall", 0.0)),
            -float(config.get("selected_segments", 0)),
        )
        candidates.append((key, config, {"fold": fold.get("fold"), "cv_segment": segment}))
    if not candidates:
        return {"enabled": False}, {"strategy": "no_enabled_train_cv_graph_config"}
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, config, meta = candidates[0]
    return dict(config), {"strategy": "best_train_cv_enabled_graph_config", **meta}


def _prediction_records_for_export(names: list[str], predictions: list[list[tuple[int, int]]]) -> list[dict]:
    return [
        {
            "video_name": name,
            "segments": [[int(start), int(end)] for start, end in segments],
        }
        for name, segments in zip(names, predictions)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", default="/home/zzy/jepa_data/segment_train_full_event_v2_h32_hybrid_hyst.pt")
    parser.add_argument("--graph-summary", default="/home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_reranker_protected_cv5.summary.json")
    parser.add_argument("--selector", default="/home/zzy/jepa_data/proposal_set_selector_full_extratrees.pkl")
    parser.add_argument("--recall-selector", default="/home/zzy/jepa_data/proposal_set_selector_full_quality_extratrees.pkl")
    parser.add_argument("--switcher-summary", default="/home/zzy/jepa_data/segment_train_full_event_v2_prediction_set_switcher_strict.summary.json")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--predictions-out", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    names = [record.name for record in records]
    labels = [record.labels for record in records]
    model, mean, std, checkpoint = _load_model(args.checkpoint, args.device)
    checkpoint_metrics = checkpoint.get("metrics", {})
    mainline_outputs, mainline_predictions = _mainline_outputs(
        model,
        records,
        mean,
        std,
        args.device,
        checkpoint_metrics,
        feature_names,
    )
    mainline_params = checkpoint_metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1})
    mainline_metrics = evaluate_records(mainline_outputs, mainline_params)

    base_proposals, base_candidates, base_scores, base_selector_info = _proposal_predictions(args.selector, records, feature_names)
    graph_summary = _read_json(args.graph_summary)
    graph_config, graph_selection = _select_frozen_graph_config(graph_summary)
    graph_selected = _graph_predictions(mainline_predictions, base_candidates, base_scores, graph_config)
    protected = graph_config.get("protected", {})
    if graph_config.get("enabled", False) and protected.get("enabled", False):
        graph_base_predictions = fuse_protected_segment_predictions(
            mainline_predictions,
            graph_selected,
            max_mainline_iou=float(protected["max_mainline_iou"]),
            selector_nms_iou=protected.get("selector_nms_iou"),
        )
    else:
        graph_base_predictions = mainline_predictions
    graph_base_metrics = evaluate_fused_predictions(graph_base_predictions, labels)

    recall_proposals, recall_candidates, recall_scores, recall_selector_info = _proposal_predictions(args.recall_selector, records, feature_names)
    recall_selected = _graph_predictions(mainline_predictions, recall_candidates, recall_scores, graph_config)
    if graph_config.get("enabled", False) and protected.get("enabled", False):
        recall_predictions = fuse_protected_segment_predictions(
            mainline_predictions,
            recall_selected,
            max_mainline_iou=float(protected["max_mainline_iou"]),
            selector_nms_iou=protected.get("selector_nms_iou"),
        )
    else:
        recall_predictions = mainline_predictions
    recall_metrics = evaluate_fused_predictions(recall_predictions, labels)

    switcher = _read_json(args.switcher_summary)
    switch_configs = [fold.get("config", {}) for fold in switcher.get("folds", [])]
    selected_configs = [config for config in switch_configs if config.get("enabled", False)]
    if selected_configs:
        # Strict CV selected no switch on every fold for the frozen best. If a future
        # frozen artifact enables switching, use a deterministic majority config.
        config = sorted(selected_configs, key=lambda item: json.dumps(item, sort_keys=True))[0]
    else:
        config = {"enabled": False, "source": "strict_cv_all_folds_disabled"}
    final_predictions = apply_set_switch_config(graph_base_predictions, recall_predictions, config)
    final_metrics = evaluate_fused_predictions(final_predictions, labels)

    summary = {
        "task": "binary_error_segment_localization_heldout",
        "protocol": "frozen_full_train_artifacts_no_heldout_tuning",
        "data_dir": str(args.data_dir),
        "n_videos": len(records),
        "frames": int(sum(len(record.labels) for record in records)),
        "positive_frames": int(sum(record.labels.sum() for record in records)),
        "checkpoint": str(args.checkpoint),
        "selector": base_selector_info,
        "recall_selector": recall_selector_info,
        "graph_config": graph_config,
        "graph_config_selection": graph_selection,
        "switch_config": config,
        "mainline": mainline_metrics,
        "graph_base": graph_base_metrics,
        "recall_variant": recall_metrics,
        "final": final_metrics,
    }
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.predictions_out:
        pred_out = Path(args.predictions_out)
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        pred_out.write_text(
            json.dumps(
                {
                    "summary": str(out),
                    "predictions": _prediction_records_for_export(names, final_predictions),
                    "mainline_predictions": _prediction_records_for_export(names, mainline_predictions),
                    "graph_base_predictions": _prediction_records_for_export(names, graph_base_predictions),
                    "recall_predictions": _prediction_records_for_export(names, recall_predictions),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(json.dumps(summary["final"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
