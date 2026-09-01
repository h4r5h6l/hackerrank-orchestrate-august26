#!/usr/bin/env python3
"""Flatten labeled_seed.csv into an SDV-ready scalar feature table.

Reads dataset/labeled_seed.csv, re-parses the 9 context_* JSON blobs per row,
and emits a wide, all-scalar table (numeric / categorical / bool) with the
action + message_type labels and a synthetic integer primary key.

Usage:
    python code/features.py \
        --input dataset/labeled_seed.csv \
        --output code/generated_data/sdv_ready_seed.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any

SECTIONS = (
    "context_message",
    "context_sender",
    "context_conversation",
    "context_user",
    "context_business",
    "context_history",
    "context_daily_load",
    "context_extracted",
)


# --------------------------------------------------------------------------- #
# Coercion helpers (missing / malformed -> deterministic defaults)
# --------------------------------------------------------------------------- #

def _to_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _cat(value, default="none"):
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _parse_clock(value):
    """Parse 'HH:MM' into (hour, minute); return None if malformed."""
    try:
        hour_s, minute_s = value.strip().split(":")
        hour, minute = int(hour_s), int(minute_s)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute
    except (ValueError, AttributeError):
        pass
    return None


def _minutes_of_day(hour, minute):
    return hour * 60 + minute


def _in_quiet_hours(quiet, hour, minute):
    """True if time-of-day falls inside 'HH:MM-HH:MM', handling midnight wrap."""
    span = (quiet or "").split("-")
    if len(span) != 2:
        return False
    start = _parse_clock(span[0])
    end = _parse_clock(span[1])
    if start is None or end is None:
        return False
    t = _minutes_of_day(hour, minute)
    s = _minutes_of_day(*start)
    e = _minutes_of_day(*end)
    if s <= e:
        return s <= t < e
    return t >= s or t < e  # window wraps midnight


# --------------------------------------------------------------------------- #
# Feature flattening
# --------------------------------------------------------------------------- #

def flatten_row(row):
    ctx = {name: json.loads(row.get(name) or "{}") for name in SECTIONS}

    msg = ctx["context_message"]
    sender = ctx["context_sender"]
    convo = ctx["context_conversation"]
    user = ctx["context_user"]
    biz = ctx["context_business"]
    hist = ctx["context_history"]
    extracted = ctx["context_extracted"]

    membership = convo.get("membership") or {}
    group_record = convo.get("record") or {}
    sender_record = sender.get("record") or {}
    user_record = user.get("record") or {}
    biz_record = biz.get("record") or {}
    relationship = biz.get("relationship") or {}
    prev_interactions = hist.get("previous_interactions") or {}

    timestamp = _cat(msg.get("timestamp"), "")
    timestamp_hour = _to_int(timestamp[11:13], 0) if len(timestamp) >= 13 else 0
    timestamp_minute = _to_int(timestamp[14:16], 0) if len(timestamp) >= 16 else 0

    quiet_hours = _cat(user.get("quiet_hours"), "")
    media_type = _cat(msg.get("media_type"))
    domain_used = _cat(biz_record.get("domain_used_by_sender"), "")
    official_domain = _cat(biz_record.get("official_domain"), "")

    f = {}

    # ---- message ----
    f["media_type"] = media_type
    f["has_media"] = media_type != "none"
    f["text_len"] = len(_cat(msg.get("text"), ""))
    f["timestamp_hour"] = timestamp_hour
    f["timestamp_minute"] = timestamp_minute

    # ---- extracted signals ----
    f["asks_for_reply"] = _to_bool(extracted.get("asks_for_reply"))
    f["deadline"] = _to_bool(extracted.get("deadline"))
    f["has_direct_mention"] = _to_bool(extracted.get("has_direct_mention"))
    f["is_greeting"] = _to_bool(extracted.get("is_greeting"))
    f["opt_out"] = _to_bool(extracted.get("opt_out"))
    f["otp"] = _to_bool(extracted.get("otp"))
    f["payment"] = _to_bool(extracted.get("payment"))
    f["promotional"] = _to_bool(extracted.get("promotional"))
    f["suspicious_link"] = _to_bool(extracted.get("suspicious_link"))
    f["is_forwarded"] = _to_bool(extracted.get("is_forwarded"))
    f["has_url"] = bool(extracted.get("urls"))
    f["n_mentions"] = len(extracted.get("mentions") or [])

    # ---- conversation / group ----
    chat_type = _cat(convo.get("chat_type"))
    f["chat_type"] = chat_type
    f["is_group"] = chat_type == "group"
    f["recent_messages_count"] = len(convo.get("recent_messages") or [])
    f["group_muted_by_user"] = _to_bool(membership.get("group_muted_by_user"))
    f["membership_role"] = _cat(membership.get("role"))
    f["membership_messages_read_30d"] = _to_int(membership.get("messages_read_30d"))
    f["membership_messages_sent_30d"] = _to_int(membership.get("messages_sent_30d"))
    f["membership_replies_sent_30d"] = _to_int(membership.get("replies_sent_30d"))
    f["membership_notifications_dismissed_30d"] = _to_int(
        membership.get("notifications_dismissed_30d"))
    f["group_member_count"] = _to_int(group_record.get("member_count"))
    f["group_admin_count"] = _to_int(group_record.get("admin_count"))
    f["group_messages_30d"] = _to_int(group_record.get("messages_30d"))
    f["group_type"] = _cat(group_record.get("group_type"))

    # ---- sender ----
    f["sender_is_business"] = _to_bool(sender.get("is_business"))
    f["sender_is_verified"] = _to_bool(sender.get("is_verified"))
    f["sender_type"] = _cat(sender.get("type"))
    f["sender_messages_opened_30d"] = _to_int(sender_record.get("messages_opened_30d"))
    f["sender_messages_replied_30d"] = _to_int(sender_record.get("messages_replied_30d"))
    f["sender_messages_reported_30d"] = _to_int(sender_record.get("messages_reported_30d"))
    f["sender_notifications_dismissed_30d"] = _to_int(
        sender_record.get("notifications_dismissed_30d"))

    # ---- user (receiver) ----
    f["user_messages_opened_30d"] = _to_int(user_record.get("messages_opened_30d"))
    f["user_messages_replied_30d"] = _to_int(user_record.get("messages_replied_30d"))
    f["user_messages_reported_30d"] = _to_int(user_record.get("messages_reported_30d"))
    f["user_notifications_dismissed_30d"] = _to_int(
        user_record.get("notifications_dismissed_30d"))
    f["has_quiet_hours"] = bool(quiet_hours and quiet_hours != "none")
    f["is_in_quiet_hours"] = _in_quiet_hours(quiet_hours, timestamp_hour, timestamp_minute)

    # ---- business ----
    f["is_business"] = bool(biz_record)
    f["business_is_verified"] = _to_bool(biz_record.get("verified"))
    f["business_account_age_days"] = _to_int(biz_record.get("account_age_days"))
    f["business_domain_age_days"] = _to_int(biz_record.get("domain_used_by_sender_age_days"))
    f["business_messages_sent_30d"] = _to_int(biz_record.get("messages_sent_30d"))
    f["business_user_reports_30d"] = _to_int(biz_record.get("user_reports_30d"))
    f["business_category"] = _cat(biz_record.get("category"))
    f["domain_matches_official"] = domain_used != "" and domain_used == official_domain
    f["purchase_history_count"] = _to_int(biz.get("purchase_history"))
    f["rel_activity_count_180d"] = _to_int(relationship.get("activity_count_180d"))
    f["rel_messages_opened_30d"] = _to_int(relationship.get("messages_opened_30d"))
    f["rel_messages_dismissed_30d"] = _to_int(relationship.get("messages_dismissed_30d"))
    f["rel_messages_replied_30d"] = _to_int(relationship.get("messages_replied_30d"))
    f["rel_allows_promotions"] = _cat(relationship.get("allows_promotions"))
    f["rel_opted_out"] = bool(_cat(relationship.get("promotions_opted_out_at"), ""))
    f["subscription"] = _cat(biz.get("subscription"))

    # ---- history ----
    f["hist_message_opened"] = _to_int(prev_interactions.get("message_opened"))
    f["hist_message_replied"] = _to_int(prev_interactions.get("message_replied"))
    f["hist_message_reported"] = _to_int(prev_interactions.get("message_reported"))
    f["hist_muted_after_message"] = _to_int(prev_interactions.get("muted_after_message"))
    f["hist_notification_dismissed"] = _to_int(
        prev_interactions.get("notification_dismissed"))
    f["similar_messages_count"] = len(hist.get("similar_messages") or [])

    # ---- passthrough numeric ----
    f["priority"] = _to_float(row.get("priority"))
    f["confidence"] = _to_float(row.get("confidence"))

    # ---- labels & key ----
    f["action"] = _cat(row.get("action"))
    f["message_type"] = _cat(row.get("message_type"))
    f["message_id"] = _cat(row.get("message_id"))

    return f


def flatten_features(seed_rows):
    """Flatten every seed row; deterministic column order across all rows."""
    if not seed_rows:
        return []
    flat_rows = [flatten_row(row) for row in seed_rows]
    columns = []
    for flat in flat_rows:
        for key in flat:
            if key not in columns:
                columns.append(key)
    columns = ["sdv_primary_key"] + columns
    for index, flat in enumerate(flat_rows):
        flat["sdv_primary_key"] = index
    return [{col: flat[col] for col in columns} for flat in flat_rows]


def _read_seed_rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_features(rows, path):
    if not rows:
        raise SystemExit("No rows to write.")
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Flatten labeled_seed.csv for SDV")
    parser.add_argument("--input", default="dataset/labeled_seed.csv")
    parser.add_argument("--output", default="code/generated_data/sdv_ready_seed.csv")
    args = parser.parse_args(argv)

    seed_rows = _read_seed_rows(args.input)
    flat_rows = flatten_features(seed_rows)
    _write_features(flat_rows, args.output)

    print("Wrote %d rows x %d columns -> %s"
          % (len(flat_rows), len(flat_rows[0]), args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())


