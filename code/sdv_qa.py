#!/usr/bin/env python3
"""QA the synthetic seed table against the real one (threshold-gated).

Checks:
  1. Label distribution parity for action and message_type.
  2. Derived-constraint consistency: the joint invariants features.py
     guarantees in the real table (chat_type / sender_type / is_group,
     media flags, group-only and business-only columns) must hold exactly
     in the synthetic table. Each rule is first asserted against the real
     table so the gate stays honest.
  3. Numeric correlation-matrix similarity (mean off-diagonal |delta rho|).
  4. Joint P(action | message_type) similarity.
  5. Leakage: no real message_id reuse; unique synthetic primary keys.
  6. Completeness: no missing cells in the synthetic table.

Writes a JSON report (default code/generated_data/sdv_qa_report.json) and
prints a PASS/FAIL summary. Exit code 0 when every gate passes, else 1.

Usage:
    python code/sdv_qa.py \
        --real code/generated_data/sdv_ready_seed.csv \
        --synthetic code/generated_data/sdv_synthetic.csv \
        --report code/generated_data/sdv_qa_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DELTA_SHARE_MAX = 0.05   # label distribution parity
DELTA_JOINT_MAX = 0.10   # P(action | message_type) parity
CORR_MEAN_MAX = 0.12     # mean |corr delta| over numeric off-diagonal pairs
TOP_WORST_PAIRS = 5

GROUP_COUNT_COLS = ("group_member_count", "group_admin_count", "group_messages_30d")
MEMBERSHIP_COUNT_COLS = (
    "membership_messages_read_30d",
    "membership_messages_sent_30d",
    "membership_replies_sent_30d",
    "membership_notifications_dismissed_30d",
)
NON_BUSINESS_COUNT_COLS = (
    "business_account_age_days",
    "business_domain_age_days",
    "business_messages_sent_30d",
    "business_user_reports_30d",
    "purchase_history_count",
    "rel_activity_count_180d",
    "rel_messages_opened_30d",
    "rel_messages_dismissed_30d",
    "rel_messages_replied_30d",
)
NON_BUSINESS_FLAG_COLS = ("rel_opted_out", "domain_matches_official", "business_is_verified")
NON_BUSINESS_CAT_COLS = ("business_category", "subscription", "rel_allows_promotions")


def _constraint_exprs():
    """Return (name, expr) pairs; expr(df) is True where the rule holds."""
    return [
        ("is_group == (chat_type == 'group')",
         lambda df: df["is_group"] == (df["chat_type"] == "group")),
        ("is_business == (chat_type == 'business')",
         lambda df: df["is_business"] == (df["chat_type"] == "business")),
        ("sender_is_business == (chat_type == 'business')",
         lambda df: df["sender_is_business"] == (df["chat_type"] == "business")),
        ("sender_type == chat_type",
         lambda df: df["sender_type"] == df["chat_type"]),
        ("has_media == (media_type != 'none')",
         lambda df: df["has_media"] == (df["media_type"] != "none")),
        ("n_mentions == int(has_direct_mention)",
         lambda df: df["n_mentions"].astype(int) == df["has_direct_mention"].astype(int)),
        ("non-group membership_role == 'none'",
         lambda df: (df["chat_type"] == "group") | (df["membership_role"] == "none")),
        ("group membership_role != 'none'",
         lambda df: (df["chat_type"] != "group") | (df["membership_role"] != "none")),
        ("non-group group_type == 'none'",
         lambda df: (df["chat_type"] == "group") | (df["group_type"] == "none")),
        ("non-group group_muted_by_user is False",
         lambda df: (df["chat_type"] == "group") | (~df["group_muted_by_user"].astype(bool))),
        ("non-group group counts are 0",
         lambda df: (df["chat_type"] == "group")
         | (df[list(GROUP_COUNT_COLS)].fillna(0).astype(int).sum(axis=1) == 0)),
        ("non-group membership counts are 0",
         lambda df: (df["chat_type"] == "group")
         | (df[list(MEMBERSHIP_COUNT_COLS)].fillna(0).astype(int).sum(axis=1) == 0)),
        ("non-business business flags are False",
         lambda df: (df["chat_type"] == "business")
         | (~df[list(NON_BUSINESS_FLAG_COLS)].astype(bool).any(axis=1))),
        ("non-business business cats are 'none'",
         lambda df: (df["chat_type"] == "business")
         | ((df[list(NON_BUSINESS_CAT_COLS)].astype(str) != "none").sum(axis=1) == 0)),
        ("non-business business counts are 0",
         lambda df: (df["chat_type"] == "business")
         | (df[list(NON_BUSINESS_COUNT_COLS)].fillna(0).astype(int).sum(axis=1) == 0)),
    ]


def check_label_parity(real, syn, column):
    real_share = real[column].value_counts(normalize=True)
    syn_share = syn[column].value_counts(normalize=True)
    classes = sorted(set(real_share.index) | set(syn_share.index))
    deltas = {c: abs(float(real_share.get(c, 0.0)) - float(syn_share.get(c, 0.0))) for c in classes}
    max_delta = max(deltas.values(), default=0.0)
    missing = [str(c) for c in real_share.index if float(syn_share.get(c, 0.0)) == 0.0]
    return {
        "passed": bool(max_delta <= DELTA_SHARE_MAX and not missing),
        "max_abs_share_delta": round(max_delta, 4),
        "missing_classes_in_synthetic": missing,
        "per_class_share_delta": {str(k): round(v, 4) for k, v in sorted(deltas.items())},
    }


def check_constraints(real, syn):
    constraints = []
    all_passed = True
    for name, expr in _constraint_exprs():
        try:
            real_bad = int((~expr(real)).sum())
        except (KeyError, TypeError, ValueError):
            constraints.append({"name": name, "status": "skipped", "detail": "columns missing"})
            continue
        if real_bad:
            all_passed = False
            constraints.append({"name": name, "status": "invalid-rule",
                                "detail": f"real table violates the rule in {real_bad} rows"})
            continue
        syn_bad = int((~expr(syn)).sum())
        passed = syn_bad == 0
        all_passed = all_passed and passed
        constraints.append({"name": name, "status": "passed" if passed else "failed",
                            "real_violations": real_bad, "synthetic_violations": syn_bad})
    return {"passed": bool(all_passed), "constraints": constraints}


def check_correlations(real, syn):
    exclude = {"sdv_primary_key", "message_id", "action", "message_type"}
    numeric = [
        c for c in real.columns
        if c not in exclude and c in syn.columns
        and pd.api.types.is_numeric_dtype(real[c]) and real[c].dtype != bool
        and pd.api.types.is_numeric_dtype(syn[c]) and syn[c].dtype != bool
    ]
    if len(numeric) < 2:
        return {"passed": True, "mean_abs_delta": 0.0, "detail": "not enough numeric columns"}
    diff = (real[numeric].corr() - syn[numeric].corr()).abs().to_numpy()
    off = ~np.eye(len(numeric), dtype=bool)
    mean_delta = float(diff[off].mean())
    max_delta = float(diff[off].max())
    pairs = sorted(
        ((numeric[i], numeric[j], float(diff[i, j]))
         for i in range(len(numeric)) for j in range(i + 1, len(numeric))),
        key=lambda item: -item[2],
    )
    return {
        "passed": bool(mean_delta <= CORR_MEAN_MAX),
        "mean_abs_delta": round(mean_delta, 4),
        "max_abs_delta": round(max_delta, 4),
        "numeric_columns_compared": len(numeric),
        "worst_pairs": [
            {"pair": f"{a} vs {b}", "abs_delta": round(d, 4)}
            for a, b, d in pairs[:TOP_WORST_PAIRS]
        ],
    }


def check_joint(real, syn):
    joint_real = pd.crosstab(real["message_type"], real["action"], normalize="index")
    joint_syn = pd.crosstab(syn["message_type"], syn["action"], normalize="index")
    per_type = {}
    max_delta = 0.0
    for mtype in joint_real.index:
        if mtype not in joint_syn.index:
            return {"passed": False, "detail": f"message_type '{mtype}' missing in synthetic"}
        delta = float((joint_real.loc[mtype] - joint_syn.loc[mtype]).abs().max())
        per_type[str(mtype)] = round(delta, 4)
        max_delta = max(max_delta, delta)
    return {
        "passed": bool(max_delta <= DELTA_JOINT_MAX),
        "max_abs_delta": round(max_delta, 4),
        "per_message_type": per_type,
    }


def check_leakage(real, syn, history_ids, message_ids):
    syn_ids = set(syn["message_id"].astype(str))
    overlaps = {
        "seed": len(syn_ids & set(real["message_id"].astype(str))),
        "message_history": len(syn_ids & history_ids),
        "messages": len(syn_ids & message_ids),
    }
    duplicate_keys = int(syn["sdv_primary_key"].duplicated().sum())
    feature_cols = [c for c in real.columns
                    if c not in ("sdv_primary_key", "message_id", "action", "message_type")]
    exact_feature_dupes = int(syn.merge(
        real[feature_cols].drop_duplicates(), on=feature_cols, how="inner").shape[0])
    return {
        "passed": bool(all(v == 0 for v in overlaps.values()) and duplicate_keys == 0),
        "id_overlaps": overlaps,
        "duplicate_primary_keys": duplicate_keys,
        "exact_feature_duplicates_vs_real": exact_feature_dupes,
    }


def check_completeness(syn):
    missing = int(syn.isna().sum().sum())
    return {"passed": missing == 0, "missing_cells": missing}


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="QA the SDV synthetic seed table")
    parser.add_argument("--real", default=str(root / "code/generated_data/sdv_ready_seed.csv"))
    parser.add_argument("--synthetic", default=str(root / "code/generated_data/sdv_synthetic.csv"))
    parser.add_argument("--history", default=str(root / "dataset/message_history.csv"))
    parser.add_argument("--messages", default=str(root / "dataset/messages.csv"))
    parser.add_argument("--report", default=str(root / "code/generated_data/sdv_qa_report.json"))
    args = parser.parse_args(argv)

    real = pd.read_csv(args.real)
    syn = pd.read_csv(args.synthetic)
    history_ids = set(pd.read_csv(args.history, usecols=["message_id"])["message_id"].astype(str))
    message_ids = set(pd.read_csv(args.messages, usecols=["message_id"])["message_id"].astype(str))

    checks = {
        "label_parity_action": check_label_parity(real, syn, "action"),
        "label_parity_message_type": check_label_parity(real, syn, "message_type"),
        "derived_constraints": check_constraints(real, syn),
        "correlation_similarity": check_correlations(real, syn),
        "joint_action_given_type": check_joint(real, syn),
        "leakage": check_leakage(real, syn, history_ids, message_ids),
        "completeness": check_completeness(syn),
    }

    report = {"all_passed": all(check["passed"] for check in checks.values()), "checks": checks}
    parent = os.path.dirname(os.path.abspath(args.report))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    for name, result in checks.items():
        print(f"[{'PASS' if result['passed'] else 'FAIL'}] {name}")
        if name == "derived_constraints":
            for constraint in result["constraints"]:
                if constraint["status"] != "passed":
                    detail = constraint.get("synthetic_violations", constraint.get("detail"))
                    print(f"    - {constraint['name']}: {constraint['status']} ({detail})")
        elif "max_abs_share_delta" in result:
            print(f"    max |share delta| = {result['max_abs_share_delta']}")
        elif "mean_abs_delta" in result:
            print(f"    mean |corr delta| = {result['mean_abs_delta']} (max {result['max_abs_delta']})")
        elif "max_abs_delta" in result:
            print(f"    max |delta| = {result['max_abs_delta']}")
        elif "id_overlaps" in result:
            print(f"    id overlaps = {result['id_overlaps']}, duplicate keys = {result['duplicate_primary_keys']}")
        elif "missing_cells" in result:
            print(f"    missing cells = {result['missing_cells']}")

    print()
    print("ALL CHECKS PASSED" if report["all_passed"] else "QA FAILURES PRESENT")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
