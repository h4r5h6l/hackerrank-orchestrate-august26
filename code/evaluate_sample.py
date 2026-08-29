"""Evaluate the deterministic router against human-labeled sample_messages.csv.

Reports accuracy, per-class precision/recall/F1 (macro), and confusion-matrix
heatmaps for both predicted `action` and predicted `message_type`.

Heatmaps are rendered as colour-coded HTML (code/evaluation/*.html) and as a
plain-text matrix on the console. A per-sample CSV report is also written to
code/evaluation/sample_predictions.csv.

Run:  python3 code/evaluate_sample.py   (from the repo root)
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from classification import classify_message_type
from context import build_context
from seed_pipeline import calculate_priority, choose_action, normalize_message

CLASSES = (
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
)
ACTIONS = ("notify", "digest", "mute")

DATA_NAMES = (
    "users",
    "groups",
    "business_accounts",
    "group_members",
    "user_business_history",
    "message_history",
    "message_events",
    "daily_notification_summary",
    "images",
    "voice_notes",
)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_predictions(samples, data):
    out = []
    for row in samples:
        ctx = build_context(normalize_message(row), data)
        pred_type = classify_message_type(ctx)
        priority = calculate_priority(ctx, pred_type)
        pred_action, conf = choose_action(ctx, pred_type, priority)
        out.append(
            {
                "message_id": row.get("message_id", ""),
                "pred_type": pred_type,
                "true_type": row.get("message_type", "").strip().lower(),
                "pred_action": pred_action,
                "true_action": row.get("action", "").strip().lower(),
                "priority": priority,
                "confidence": conf,
                "true_confidence": row.get("confidence", ""),
            }
        )
    return out


def confusion_counts(preds, truths, classes):
    index = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    for p, t in zip(preds, truths):
        if p in index and t in index:
            cm[index[t]][index[p]] += 1
    return cm


def class_metrics(preds, truths, classes):
    present = [c for c in classes if c in set(truths) | set(preds)]
    tp = Counter(); fp = Counter(); fn = Counter()
    for p, t in zip(preds, truths):
        for c in classes:
            if c == t and c == p:
                tp[c] += 1
            elif c == p and c != t:
                fp[c] += 1
            elif c == t and c != p:
                fn[c] += 1

    def f1(p, r):
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    rows = []
    for c in present:
        P = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        R = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        rows.append((c, tp[c], fp[c], fn[c], P, R, f1(P, R)))
    support = {c: Counter(truths).get(c, 0) for c in present}
    pv = [x[4] for x in rows]; rv = [x[5] for x in rows]; fv = [x[6] for x in rows]
    macro = ((sum(pv) / len(pv), sum(rv) / len(rv), sum(fv) / len(fv)) if pv else (0.0, 0.0, 0.0))
    return rows, macro, support


def metrics_summary(preds, truths, classes, label):
    rows, macro, support = class_metrics(preds, truths, classes)
    correct = sum(1 for p, t in zip(preds, truths) if p == t)
    acc = correct / len(preds)
    print(f"\n===== {label} ({len(preds)} samples) =====")
    print(f"Accuracy: {acc:.3f}  ({correct}/{len(preds)})")
    print(f"{'class':<16}{'support':>8}{'tp':>4}{'fp':>4}{'fn':>4}{'prec':>8}{'rec':>7}{'f1':>7}")
    for c, tp, fp, fn, P, R, F in rows:
        print(f"{c:<16}{support[c]:>8}{tp:>4}{fp:>4}{fn:>4}{P:>8.3f}{R:>7.3f}{F:>7.3f}")
    mp, mr, mf = macro
    print(f"Macro avg: precision={mp:.3f} recall={mr:.3f} f1={mf:.3f}")
    return acc, macro, rows


def ascii_heatmap(cm, classes, label):
    maxv = max((max(r) for r in cm), default=0) or 1
    w = max(len(c) for c in classes) + 2
    def cell(v):
        g = v / maxv
        if g >= 0.75:
            return "##" + str(v) + " " * (w - len(str(v)))
        if g >= 0.5:
            return "++" + str(v) + " " * (w - len(str(v)))
        if g >= 0.25:
            return "~~" + str(v) + " " * (w - len(str(v)))
        if g > 0:
            return ".." + str(v) + " " * (w - len(str(v)))
        return "  " + str(v) + " " * (w - len(str(v)))
    print(f"\n{label}: (rows=truth, cols=prediction)")
    print(" " * w + "".join(f"{c:>{w + 2}}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:>{w}}" + "".join(cell(v) for v in cm[i]))
    print("Key: ## high, ++ med-high, ~~ med, .. low, blank zero")


def html_heatmap(cm, classes, label, out_path: Path):
    maxv = max((max(r) for r in cm), default=0) or 1
    def bg(v):
        t = v / maxv
        if t <= 0:
            return ""
        return f"background-color:rgba(220,20,60,{0.15 + 0.85 * t:.2f});color:#fff"
    rows_html = ""
    for i, c in enumerate(classes):
        cells = "".join(
            f'<td style="{bg(v)};text-align:center;padding:8px 10px;border:1px solid #ccc">{v}</td>'
            for v in cm[i]
        )
        rows_html += f'<tr><th style="text-align:left;padding:8px">{c}</th>{cells}</tr>'
    head = "".join(f'<th style="padding:8px">{c}</th>' for c in classes)
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:sans-serif}table{border-collapse:collapse}"
        "th,td{border:1px solid #ccc}caption{font-weight:bold;font-size:1.1em;padding:8px}</style>"
        f"</head><body><table><caption>{label}</caption>"
        f"<tr><th>truth\\pred</th>{head}</tr>{rows_html}</table></body></html>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    print(f"Heatmap written to {out_path}")


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    dataset_dir = repo / "dataset"
    ev_dir = repo / "code" / "evaluation"

    data = {name: _load_rows(dataset_dir / f"{name}.csv") for name in DATA_NAMES}
    samples = _load_rows(dataset_dir / "sample_messages.csv")
    results = run_predictions(samples, data)

    rows = [{k: r[k] for k in ("message_id", "pred_type", "true_type", "pred_action", "true_action", "priority", "confidence", "true_confidence")} for r in results]

    pred_types = [r["pred_type"] for r in results]
    true_types = [r["true_type"] for r in results]
    pred_actions = [r["pred_action"] for r in results]
    true_actions = [r["true_action"] for r in results]

    acc_a, macro_a, _ = metrics_summary(pred_actions, true_actions, ACTIONS, "ACTION")
    cm_a = confusion_counts(pred_actions, true_actions, ACTIONS)
    ascii_heatmap(cm_a, ACTIONS, "ACTION")
    html_heatmap(cm_a, ACTIONS, "ACTION confusion matrix", ev_dir / "action_confusion.html")

    acc_t, macro_t, _ = metrics_summary(pred_types, true_types, CLASSES, "MESSAGE_TYPE")
    cm_t = confusion_counts(pred_types, true_types, CLASSES)
    html_heatmap(cm_t, CLASSES, "MESSAGE_TYPE confusion matrix", ev_dir / "message_type_confusion.html")

    print("\n===== OVERALL =====")
    print(f"Action accuracy: {acc_a:.3f} ({100 * acc_a:.1f}%)  macro-F1: {macro_a[2]:.3f}")
    print(f"Message-type accuracy: {acc_t:.3f} ({100 * acc_t:.1f}%)  macro-F1: {macro_t[2]:.3f}")

    report_path = ev_dir / "sample_predictions.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-sample predictions written to {report_path}")


if __name__ == "__main__":
    main()
