#!/usr/bin/env python3
"""Train XGBoost action and message-type models on real + SDV synthetic rows.

Training mix: code/generated_data/sdv_synthetic.csv (10,000 distilled rows)
plus the 92 real seed rows from code/generated_data/sdv_ready_seed.csv,
upweighted (default x10) so real rows anchor the distribution. Identifiers
and rule-engine outputs (confidence, priority) are excluded from features to
avoid target leakage, so the model learns routing from message context only.

Saves:
    code/generated_data/xgb_action.json         3-class action booster
    code/generated_data/xgb_message_type.json   10-class message-type booster
    code/generated_data/xgb_spec.json           feature spec + classes + meta

Also prints a k-fold CV over the real seed rows using the production recipe
(synthetic + real-train-fold), as an honest fit sanity check.

Usage:
    python code/train_xgb.py --real-weight 10 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from xgb_common import GENERATED_DIR, build_feature_spec, save_spec, vectorize_rows


def model_params(seed: int) -> dict[str, Any]:
    return {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": 4,
        "eval_metric": "mlogloss",
    }


def fit_model(matrix, y, weights, seed: int) -> XGBClassifier:
    """Fit one multi-class booster on pre-encoded integer labels."""
    model = XGBClassifier(**model_params(seed))
    model.fit(matrix, y, sample_weight=weights)
    return model


def cv_report(train, matrix, syn_count, action_index, type_index, folds, seed, real_weight):
    """K-fold CV over the real rows; each fold trains on the production mix."""
    real_count = len(train) - syn_count
    real_offset = syn_count
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    summary = {}
    for target, index in (("action", action_index), ("message_type", type_index)):
        y = train[target].astype(str).map(index).to_numpy()
        accs, f1s = [], []
        for train_idx, valid_idx in splitter.split(np.arange(real_count)):
            fit_idx = np.concatenate([np.arange(syn_count), real_offset + train_idx])
            fit_weights = np.concatenate(
                [
                    np.ones(syn_count, dtype=np.float32),
                    np.full(len(train_idx), real_weight, dtype=np.float32),
                ]
            )
            model = fit_model(matrix[fit_idx], y[fit_idx], fit_weights, seed)
            preds = model.predict(matrix[real_offset + valid_idx])
            truth = y[real_offset + valid_idx]
            accs.append(float((preds == truth).mean()))
            f1s.append(
                float(f1_score(truth, preds, average="macro", labels=list(range(len(index))), zero_division=0))
            )
        summary[target] = {
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_folds": [round(a, 4) for a in accs],
            "macro_f1_mean": round(float(np.mean(f1s)), 4),
        }
        print(
            f"CV {target}: accuracy {summary[target]['accuracy_mean']} "
            f"(folds {summary[target]['accuracy_folds']}), macro-F1 {summary[target]['macro_f1_mean']}"
        )
    return summary


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train XGBoost router models")
    parser.add_argument("--synthetic", default=str(root / "code/generated_data/sdv_synthetic.csv"))
    parser.add_argument("--real", default=str(root / "code/generated_data/sdv_ready_seed.csv"))
    parser.add_argument("--real-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args(argv)

    syn = pd.read_csv(args.synthetic)
    real = pd.read_csv(args.real)
    train = pd.concat([syn, real], ignore_index=True)
    weights = np.concatenate(
        [
            np.ones(len(syn), dtype=np.float32),
            np.full(len(real), args.real_weight, dtype=np.float32),
        ]
    )
    print(
        f"Training rows: {len(syn)} synthetic + {len(real)} real "
        f"(weight x{args.real_weight:g}) = {len(train)}"
    )

    spec = build_feature_spec(train)
    one_hot = sum(len(values) for values in spec["categorical"].values())
    n_features = one_hot + len(spec["boolean"]) + len(spec["numeric"])
    print(
        f"Features: {len(spec['categorical'])} categorical "
        f"(one-hot -> {one_hot}), {len(spec['boolean'])} boolean, "
        f"{len(spec['numeric'])} numeric => {n_features} columns"
    )

    matrix = vectorize_rows(train.to_dict("records"), spec)
    action_classes = sorted(train["action"].astype(str).unique())
    type_classes = sorted(train["message_type"].astype(str).unique())

    action_index = {cls: i for i, cls in enumerate(action_classes)}
    type_index = {cls: i for i, cls in enumerate(type_classes)}
    action_y = train["action"].astype(str).map(action_index).to_numpy()
    type_y = train["message_type"].astype(str).map(type_index).to_numpy()
    action_model = fit_model(matrix, action_y, weights, args.seed)
    type_model = fit_model(matrix, type_y, weights, args.seed)

    action_train_acc = float((action_model.predict(matrix) == action_y).mean())
    type_train_acc = float((type_model.predict(matrix) == type_y).mean())
    print(f"In-sample fit: action acc {action_train_acc:.4f}, message_type acc {type_train_acc:.4f}")

    print(f"Running {args.cv_folds}-fold CV over the real seed rows...")
    cv_summary = cv_report(
        train, matrix, len(syn), action_index, type_index, args.cv_folds, args.seed, args.real_weight
    )

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    action_model.get_booster().save_model(str(GENERATED_DIR / "xgb_action.json"))
    type_model.get_booster().save_model(str(GENERATED_DIR / "xgb_message_type.json"))
    meta = {
        "synthetic_rows": int(len(syn)),
        "real_rows": int(len(real)),
        "real_weight": float(args.real_weight),
        "seed": int(args.seed),
        "params": model_params(args.seed),
        "feature_count": int(n_features),
        "in_sample_accuracy": {"action": round(action_train_acc, 4), "message_type": round(type_train_acc, 4)},
        "cv": cv_summary,
    }
    spec_path = save_spec(GENERATED_DIR / "xgb_spec.json", spec, action_classes, type_classes, meta)
    print(f"Saved boosters + spec to {GENERATED_DIR} (spec: {spec_path.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
