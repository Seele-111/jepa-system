#!/usr/bin/env python3
"""Helpers for ensembling temporal segment locator predictions."""
from __future__ import annotations

import numpy as np

from train_segment_locator import _segments_from_params


def average_prediction_records(model_outputs: list[list[dict]]) -> list[dict]:
    if not model_outputs:
        return []
    expected_len = len(model_outputs[0])
    for outputs in model_outputs:
        if len(outputs) != expected_len:
            raise ValueError(f"ensemble output length mismatch: {len(outputs)} vs {expected_len}")
    averaged: list[dict] = []
    for records in zip(*model_outputs):
        name = str(records[0].get("name", ""))
        labels = np.asarray(records[0]["labels"], dtype=np.int64).reshape(-1)
        probs_list = []
        for record in records:
            if str(record.get("name", "")) != name:
                raise ValueError(f"ensemble record order mismatch: {record.get('name')} vs {name}")
            probs = np.asarray(record["probs"], dtype=np.float32).reshape(-1)
            n = min(len(labels), len(probs))
            probs_list.append(probs[:n])
            labels = labels[:n]
        n = min(len(item) for item in probs_list)
        stacked = np.stack([item[:n] for item in probs_list], axis=0)
        averaged.append(
            {
                "name": name,
                "labels": labels[:n].copy(),
                "probs": np.nan_to_num(stacked.mean(axis=0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32),
            }
        )
    return averaged


def records_to_segments(records: list[dict], params: dict) -> list[list[tuple[int, int]]]:
    return [_segments_from_params(np.asarray(record["probs"], dtype=np.float32), params) for record in records]
