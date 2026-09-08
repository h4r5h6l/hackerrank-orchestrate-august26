"""Shared helpers for XGBoost training and inference on the flattened tables.

Feature contract (shared by training and inference):
  - features come from features.flatten_row() over the 8 context sections
  - excluded from the model: identifiers (message_id, sdv_primary_key) and
    rule-engine outputs (confidence, priority) to avoid target leakage
  - categoricals are one-hot encoded against a category list frozen at
    training time; unseen categories encode as an all-zero block
  - booleans encode as 0/1, numerics as floats (missing -> 0.0)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from xgboost import Booster

from features import flatten_row

GENERATED_DIR = Path(__file__).resolve().parent / "generated_data"

EXCLUDED_COLS = (
    "sdv_primary_key",
    "message_id",
    "action",
    "message_type",
    "confidence",
    "priority",
)

SECTION_NAMES = (
    "message",
    "sender",
    "conversation",
    "user",
    "business",
    "history",
    "daily_load",
    "extracted",
)


def context_to_feature_row(
    message_id: str,
    context: dict[str, Any],
    action: str = "",
    message_type: str = "",
    priority: float = 0.0,
    confidence: float = 0.0,
) -> dict[str, Any]:
    """Bridge a live build_context() dict into a features.flatten_row() input."""
    row = {
        f"context_{name}": json.dumps(context.get(name, {}), ensure_ascii=True, sort_keys=True, default=str)
        for name in SECTION_NAMES
    }
    row.update(
        {
            "message_id": message_id,
            "action": action,
            "message_type": message_type,
            "priority": priority,
            "confidence": confidence,
        }
    )
    return flatten_row(row)


def build_feature_spec(frame) -> dict[str, Any]:
    """Derive the ordered feature specification from a training table."""
    categorical: dict[str, list[str]] = {}
    boolean: list[str] = []
    numeric: list[str] = []
    for col in frame.columns:
        if col in EXCLUDED_COLS:
            continue
        dtype = frame[col].dtype
        if dtype == object:
            categorical[col] = sorted(str(value) for value in frame[col].dropna().unique())
        elif dtype == bool:
            boolean.append(col)
        else:
            numeric.append(col)
    return {"categorical": categorical, "boolean": boolean, "numeric": numeric}


def row_to_vector(row: dict[str, Any], spec: dict[str, Any]) -> list[float]:
    """Turn one flattened feature row into the model input vector."""
    values: list[float] = []
    for col, categories in spec["categorical"].items():
        value = str(row.get(col, "none"))
        values.extend(1.0 if value == category else 0.0 for category in categories)
    for col in spec["boolean"]:
        values.append(1.0 if bool(row.get(col, False)) else 0.0)
    for col in spec["numeric"]:
        try:
            values.append(float(row.get(col, 0.0)))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def vectorize_rows(rows, spec) -> np.ndarray:
    """Vectorize an iterable of flattened feature rows."""
    return np.asarray([row_to_vector(row, spec) for row in rows], dtype=np.float32)


def save_spec(path, spec, action_classes, type_classes, training_meta):
    """Persist the feature spec, class order, and training metadata."""
    payload = {
        "feature_spec": spec,
        "action_classes": list(action_classes),
        "message_type_classes": list(type_classes),
        "training": training_meta,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_models(spec_path=None) -> dict[str, Any]:
    """Load the trained boosters together with their frozen feature spec."""
    spec_path = Path(spec_path) if spec_path else GENERATED_DIR / "xgb_spec.json"
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    action_model = Booster()
    action_model.load_model(str(GENERATED_DIR / "xgb_action.json"))
    type_model = Booster()
    type_model.load_model(str(GENERATED_DIR / "xgb_message_type.json"))
    return {
        "spec": payload["feature_spec"],
        "action_classes": payload["action_classes"],
        "message_type_classes": payload["message_type_classes"],
        "action_model": action_model,
        "type_model": type_model,
        "training": payload.get("training", {}),
    }


def predict(model: Booster, classes, matrix):
    """Return (labels, probabilities) from a multi-class booster."""
    import xgboost as xgb

    matrix = np.asarray(matrix, dtype=np.float32)
    probabilities = np.asarray(model.predict(xgb.DMatrix(matrix)))
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(len(matrix), -1)
    best = probabilities.argmax(axis=1)
    return [classes[int(i)] for i in best], probabilities
