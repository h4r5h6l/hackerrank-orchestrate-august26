#!/usr/bin/env python3
"""Generate synthetic seed rows with SDV from the flattened feature table.

Fits a synthesizer on code/generated_data/sdv_ready_seed.csv and samples new
rows for XGBoost training augmentation, then reports distribution similarity
between real and synthetic data.

Usage:
    python code/sdv_generate.py \
        --input code/generated_data/sdv_ready_seed.csv \
        --output code/generated_data/sdv_synthetic.csv \
        --num-rows 10000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer

PRIMARY_KEY = "sdv_primary_key"
ID_COLS = (PRIMARY_KEY, "message_id")
LABEL_COLS = ("action", "message_type")

# Columns that are genuinely categorical in the seed table.
CATEGORICAL_COLS = (
    "media_type",
    "chat_type",
    "membership_role",
    "group_type",
    "sender_type",
    "business_category",
    "rel_allows_promotions",
    "subscription",
)

DEFAULT_OUTPUT = "code/generated_data/sdv_synthetic.csv"
DEFAULT_INPUT = "code/generated_data/sdv_ready_seed.csv"


def load_table(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    bool_cols = [
        c for c in frame.columns
        if c not in ID_COLS and frame[c].dropna().isin([True, False]).all()
        and frame[c].dtype == object
    ]
    for col in bool_cols:
        frame[col] = frame[col].astype(bool)
    for col in frame.columns:
        if col in bool_cols or col in ID_COLS or col in LABEL_COLS:
            continue
        if col in CATEGORICAL_COLS:
            frame[col] = frame[col].astype(str)
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    return frame


def build_metadata(frame: pd.DataFrame) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    metadata.add_column(PRIMARY_KEY, sdtype="id")
    for col in frame.columns:
        if col == PRIMARY_KEY:
            continue
        if col in ID_COLS:
            metadata.add_column(col, sdtype="id")
        elif col in LABEL_COLS or col in CATEGORICAL_COLS:
            metadata.add_column(col, sdtype="categorical")
        elif frame[col].dtype == bool:
            metadata.add_column(col, sdtype="boolean")
        else:
            metadata.add_column(col, sdtype="numerical")
    metadata.set_primary_key(PRIMARY_KEY)
    return metadata


def build_synthesizer(metadata: SingleTableMetadata, model: str, seed: int):
    if model == "ctgan":
        synth = CTGANSynthesizer(metadata, epochs=300, verbose=False)
    else:
        synth = GaussianCopulaSynthesizer(metadata)
    synth.reset_sampling()
    return synth


def postprocess_integers(real: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Round + clip numeric columns that are integral in real data (count-like
    columns drift out of range under the Gaussian copula)."""
    for col in real.columns:
        if col in ID_COLS or col in LABEL_COLS or col in CATEGORICAL_COLS:
            continue
        if real[col].dtype == bool or real[col].dtype == object:
            continue
        if (real[col] == real[col].astype(int)).all():
            low, high = int(real[col].min()), int(real[col].max())
            synthetic[col] = (
                synthetic[col].round().clip(low, high).astype(int)
            )
    return synthetic


def reconcile_labels(real, synthetic, rng):
    """Restore joint structure the Gaussian copula flattens out.

    - action: resample from the empirical P(action | message_type) of real data.
    - n_mentions: re-derive from has_direct_mention (real rows satisfy
      n_mentions == int(has_direct_mention)).
    """
    # --- action | message_type ---
    conditional = (
        real.groupby("message_type")["action"]
        .value_counts(normalize=True)
        .rename("prob")
        .reset_index()
    )
    choices = {
        message_type: list(group["action"])
        for message_type, group in conditional.groupby("message_type")
    }
    weights = {
        message_type: list(group["prob"])
        for message_type, group in conditional.groupby("message_type")
    }
    fallback = list(real["action"].value_counts().index)
    actions = []
    for value in synthetic["message_type"]:
        if value in choices:
            actions.append(rng.choice(choices[value], p=weights[value]))
        else:
            actions.append(rng.choice(fallback))
    synthetic["action"] = actions

    # --- n_mentions consistency ---
    if "has_direct_mention" in synthetic.columns:
        synthetic["n_mentions"] = synthetic["has_direct_mention"].astype(int)
    return synthetic


def reconcile_derived(real: pd.DataFrame, synthetic: pd.DataFrame, rng) -> pd.DataFrame:
    """Re-derive columns from their sources so joint structure holds exactly.

    features.py derives these columns from chat_type / media_type / the
    mention flag, so every real row satisfies the invariants below; the
    copula samples them independently and breaks them. Re-derivation makes
    the synthetic table satisfy them by construction. Group rows whose
    membership_role was sampled as 'none' (impossible in real data) are
    resampled from the real role distribution, and business rows with a
    non-positive account age are resampled from real positive ages.
    """
    if "chat_type" not in synthetic.columns:
        return synthetic
    group_mask = synthetic["chat_type"] == "group"
    business_mask = synthetic["chat_type"] == "business"

    # identity / type columns derive directly from chat_type
    if "is_group" in synthetic.columns:
        synthetic["is_group"] = group_mask
    if "is_business" in synthetic.columns:
        synthetic["is_business"] = business_mask
    if "sender_is_business" in synthetic.columns:
        synthetic["sender_is_business"] = business_mask
    if "sender_type" in synthetic.columns:
        synthetic["sender_type"] = synthetic["chat_type"]

    # group-only columns default to none / 0 / False outside group chats
    for col, off in (
        ("group_type", "none"),
        ("group_muted_by_user", False),
        ("group_member_count", 0),
        ("group_admin_count", 0),
        ("group_messages_30d", 0),
        ("membership_messages_read_30d", 0),
        ("membership_messages_sent_30d", 0),
        ("membership_replies_sent_30d", 0),
        ("membership_notifications_dismissed_30d", 0),
    ):
        if col in synthetic.columns:
            synthetic.loc[~group_mask, col] = off

    # business-only columns default to none / 0 / False outside business chats
    for col, off in (
        ("business_category", "none"),
        ("subscription", "none"),
        ("rel_allows_promotions", "none"),
        ("rel_opted_out", False),
        ("domain_matches_official", False),
        ("business_is_verified", False),
        ("business_account_age_days", 0),
        ("business_domain_age_days", 0),
        ("business_messages_sent_30d", 0),
        ("business_user_reports_30d", 0),
        ("purchase_history_count", 0),
        ("rel_activity_count_180d", 0),
        ("rel_messages_opened_30d", 0),
        ("rel_messages_dismissed_30d", 0),
        ("rel_messages_replied_30d", 0),
    ):
        if col in synthetic.columns:
            synthetic.loc[~business_mask, col] = off

    # membership_role: 'none' is impossible inside a group chat -> resample
    if "membership_role" in synthetic.columns:
        synthetic.loc[~group_mask, "membership_role"] = "none"
        real_roles = real.loc[
            (real["chat_type"] == "group") & (real["membership_role"] != "none"),
            "membership_role",
        ]
        if len(real_roles):
            dist = real_roles.value_counts(normalize=True)
            bad = group_mask & (synthetic["membership_role"] == "none")
            if int(bad.sum()):
                synthetic.loc[bad, "membership_role"] = rng.choice(
                    dist.index.to_list(), size=int(bad.sum()), p=dist.to_numpy()
                )

    # business rows always carry a positive account age -> resample zeros
    if "business_account_age_days" in synthetic.columns:
        real_ages = real.loc[real["business_account_age_days"] > 0, "business_account_age_days"]
        if len(real_ages):
            bad = business_mask & (synthetic["business_account_age_days"] <= 0)
            if int(bad.sum()):
                synthetic.loc[bad, "business_account_age_days"] = rng.choice(
                    real_ages.to_numpy(), size=int(bad.sum())
                )

    # media / mention flags derive from their source columns
    if "media_type" in synthetic.columns and "has_media" in synthetic.columns:
        synthetic["has_media"] = synthetic["media_type"] != "none"
    if "n_mentions" in synthetic.columns and "has_direct_mention" in synthetic.columns:
        synthetic["n_mentions"] = synthetic["has_direct_mention"].astype(int)
    return synthetic


def distribution_report(real: pd.DataFrame, synthetic: pd.DataFrame) -> str:
    lines = ["", "=== Distribution similarity (real vs synthetic) ==="]
    numeric_cols = [
        c for c in real.columns
        if c not in ID_COLS and c not in LABEL_COLS and c not in CATEGORICAL_COLS
        and real[c].dtype != bool and real[c].dtype != object
    ]
    bool_cols = [c for c in real.columns if real[c].dtype == bool]
    cat_cols = [c for c in real.columns if c in LABEL_COLS or c in CATEGORICAL_COLS]

    lines.append(f"{'column':40s} {'real mean':>10s} {'syn mean':>10s} {'real std':>9s} {'syn std':>9s}")
    for col in numeric_cols[:15]:
        lines.append(f"{col:40s} {real[col].mean():10.2f} {synthetic[col].mean():10.2f} "
                     f"{real[col].std():9.2f} {synthetic[col].std():9.2f}")

    lines.append("")
    for col in bool_cols[:10]:
        rp = real[col].mean()
        sp = synthetic[col].mean()
        lines.append(f"{col:40s} real True%={rp:6.1f}  syn True%={sp:6.1f}")

    lines.append("")
    for col in cat_cols:
        rp = real[col].value_counts(normalize=True).round(3).to_dict()
        sp = synthetic[col].value_counts(normalize=True).round(3).to_dict()
        lines.append(f"{col}: real={rp}")
        lines.append(f"{'':>{len(col)+2}} syn={sp}")
    lines.append("")
    lines.append("=== Joint structure: P(action | message_type) ===")
    joint_real = pd.crosstab(real["message_type"], real["action"], normalize="index").round(2)
    joint_syn = pd.crosstab(synthetic["message_type"], synthetic["action"], normalize="index").round(2)
    lines.append("real:")
    lines.append(joint_real.to_string())
    lines.append("synthetic:")
    lines.append(joint_syn.to_string())
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic SDV rows")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=("gaussian", "ctgan"), default="gaussian")
    parser.add_argument("--save-model", default=None)
    args = parser.parse_args(argv)

    real = load_table(args.input)
    metadata = build_metadata(real)

    import numpy as np
    np.random.seed(args.seed)

    synthesizer = build_synthesizer(metadata, args.model, args.seed)
    synthesizer.fit(real)
    synthetic = synthesizer.sample(num_rows=args.num_rows)

    # Synthetic rows must not carry real message ids.
    synthetic["message_id"] = "synthetic"
    synthetic = postprocess_integers(real, synthetic)

    import numpy as np
    rng = np.random.default_rng(args.seed)
    synthetic = reconcile_labels(real, synthetic, rng)
    synthetic = reconcile_derived(real, synthetic, rng)
    synthetic[PRIMARY_KEY] = range(len(synthetic))

    parent = os.path.dirname(os.path.abspath(args.output))
    if parent:
        os.makedirs(parent, exist_ok=True)
    synthetic.to_csv(args.output, index=False)

    if args.save_model:
        model_parent = os.path.dirname(os.path.abspath(args.save_model))
        if model_parent:
            os.makedirs(model_parent, exist_ok=True)
        synthesizer.save(args.save_model)

    print(f"Wrote {len(synthetic)} synthetic rows x {synthetic.shape[1]} columns -> {args.output}")
    print(distribution_report(real, synthetic))
    return 0


if __name__ == "__main__":
    sys.exit(main())