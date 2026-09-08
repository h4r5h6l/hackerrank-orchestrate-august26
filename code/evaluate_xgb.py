#!/usr/bin/env python3
"""Compare the rule engine, XGBoost, and a hybrid router on solved samples.

Systems evaluated on dataset/sample_messages.csv (30 human-labeled rows,
disjoint from all training data):
  - rules:  the deterministic decide_action / classify_message_type engine
  - xgb:    XGBoost models trained on real + SDV synthetic rows
  - hybrid: the rule decision stands unless rule confidence is below a
            threshold, then the XGBoost action is used (threshold swept)

Writes confusion-matrix heatmaps and a per-sample CSV to code/evaluation/.

Usage:
    python code/evaluate_xgb.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from context import build_context
from evaluate_sample import (
    ACTIONS,
    CLASSES,
    DATA_NAMES,
    _load_rows,
    class_metrics,
    confusion_counts,
    html_heatmap,
    metrics_summary,
    run_predictions,
)
from seed_pipeline import normalize_message
from xgb_common import context_to_feature_row, load_models, predict, vectorize_rows

HYBRID_THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01)


def xgb_sample_predictions(samples, data, bundle):
    """XGBoost action/type predictions and probabilities for each sample."""
    feature_rows = []
    for sample in samples:
        message = normalize_message(sample)
        context = build_context(message, data)
        feature_rows.append(context_to_feature_row(message.get("message_id", ""), context))
    matrix = vectorize_rows(feature_rows, bundle["spec"])
    action_labels, action_probs = predict(bundle["action_model"], bundle["action_classes"], matrix)
    type_labels, type_probs = predict(bundle["type_model"], bundle["message_type_classes"], matrix)
    return action_labels, action_probs, type_labels, type_probs


def hybrid_actions(rule_actions, rule_confs, xgb_actions, threshold):
    """Rule decision stands unless rule confidence is below the threshold."""
    return [
        rule if float(conf) >= threshold else xgb
        for rule, conf, xgb in zip(rule_actions, rule_confs, xgb_actions)
    ]


def _score(preds, truths, classes):
    """Return (accuracy, macro_f1) for one system."""
    correct = sum(1 for p, t in zip(preds, truths) if p == t)
    _, macro, _ = class_metrics(preds, truths, classes)
    return correct / len(preds) if preds else 0.0, macro[2]


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate rules vs XGBoost vs hybrid")
    parser.add_argument("--samples", default=str(root / "dataset/sample_messages.csv"))
    parser.add_argument("--dataset-dir", default=str(root / "dataset"))
    parser.add_argument("--output-dir", default=str(root / "code/evaluation"))
    args = parser.parse_args(argv)

    data = {name: _load_rows(Path(args.dataset_dir) / f"{name}.csv") for name in DATA_NAMES}
    samples = _load_rows(Path(args.samples))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_models()
    rule_results = run_predictions(samples, data)
    xgb_actions, xgb_action_probs, xgb_types, xgb_type_probs = xgb_sample_predictions(samples, data, bundle)

    rule_actions = [result["pred_action"] for result in rule_results]
    true_actions = [result["true_action"] for result in rule_results]
    rule_types = [result["pred_type"] for result in rule_results]
    true_types = [result["true_type"] for result in rule_results]
    rule_confs = [result["confidence"] for result in rule_results]

    acc_rule_a, macro_rule_a = _score(rule_actions, true_actions, ACTIONS)
    acc_xgb_a, macro_xgb_a = _score(xgb_actions, true_actions, ACTIONS)
    acc_rule_t, macro_rule_t = _score(rule_types, true_types, CLASSES)
    acc_xgb_t, macro_xgb_t = _score(xgb_types, true_types, CLASSES)

    sweep = []
    for threshold in HYBRID_THRESHOLDS:
        hybrid = hybrid_actions(rule_actions, rule_confs, xgb_actions, threshold)
        acc, macro_f1 = _score(hybrid, true_actions, ACTIONS)
        sweep.append((threshold, acc, macro_f1))
    best_threshold, _, _ = max(sweep, key=lambda item: (item[1], item[2]))
    hybrid_best = hybrid_actions(rule_actions, rule_confs, xgb_actions, best_threshold)
    acc_hybrid_a, macro_hybrid_a = _score(hybrid_best, true_actions, ACTIONS)

    print("\n===== HYBRID THRESHOLD SWEEP (action) =====")
    for threshold, acc, macro_f1 in sweep:
        marker = "  <- best" if threshold == best_threshold else ""
        print(f"  rule conf >= {threshold:.2f}: accuracy {acc:.3f}, macro-F1 {macro_f1:.3f}{marker}")

    metrics_summary(rule_actions, true_actions, ACTIONS, "ACTION - RULES")
    metrics_summary(xgb_actions, true_actions, ACTIONS, "ACTION - XGBOOST")
    metrics_summary(hybrid_best, true_actions, ACTIONS, f"ACTION - HYBRID (rule conf >= {best_threshold:.2f})")
    metrics_summary(rule_types, true_types, CLASSES, "MESSAGE_TYPE - RULES")
    metrics_summary(xgb_types, true_types, CLASSES, "MESSAGE_TYPE - XGBOOST")

    heatmap_specs = (
        (rule_actions, ACTIONS, "ACTION confusion - RULES", "rules_action_confusion.html"),
        (xgb_actions, ACTIONS, "ACTION confusion - XGBOOST", "xgb_action_confusion.html"),
        (hybrid_best, ACTIONS, f"ACTION confusion - HYBRID (conf >= {best_threshold:.2f})", "hybrid_action_confusion.html"),
        (rule_types, CLASSES, "MESSAGE_TYPE confusion - RULES", "rules_type_confusion.html"),
        (xgb_types, CLASSES, "MESSAGE_TYPE confusion - XGBOOST", "xgb_type_confusion.html"),
    )
    for preds, classes, label, filename in heatmap_specs:
        html_heatmap(confusion_counts(preds, true_actions if classes is ACTIONS else true_types, classes), classes, label, output_dir / filename)

    rows = []
    for i, result in enumerate(rule_results):
        rows.append(
            {
                "message_id": result["message_id"],
                "true_action": result["true_action"],
                "rule_action": result["pred_action"],
                "rule_confidence": result["confidence"],
                "xgb_action": xgb_actions[i],
                "xgb_action_confidence": round(float(xgb_action_probs[i].max()), 4),
                "hybrid_action": hybrid_best[i],
                "true_type": result["true_type"],
                "rule_type": result["pred_type"],
                "xgb_type": xgb_types[i],
                "xgb_type_confidence": round(float(xgb_type_probs[i].max()), 4),
            }
        )
    comparison_path = output_dir / "system_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== DECISION GATE =====")
    print(
        f"action: rules {acc_rule_a:.3f} (macro-F1 {macro_rule_a:.3f}) | "
        f"xgb {acc_xgb_a:.3f} ({macro_xgb_a:.3f}) | "
        f"hybrid {acc_hybrid_a:.3f} ({macro_hybrid_a:.3f} @ conf>={best_threshold:.2f})"
    )
    print(
        f"message_type: rules {acc_rule_t:.3f} ({macro_rule_t:.3f}) | "
        f"xgb {acc_xgb_t:.3f} ({macro_xgb_t:.3f})"
    )
    if acc_hybrid_a > acc_rule_a and macro_hybrid_a >= macro_rule_a and acc_xgb_t >= acc_rule_t:
        print("RECOMMENDATION: adopt the hybrid router (strictly better than rules alone).")
    else:
        print(
            "RECOMMENDATION: keep the rule engine as the production router; "
            "XGBoost ships as a calibrated second opinion."
        )
    print(f"Per-sample comparison written to {comparison_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
