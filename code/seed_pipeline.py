"""Deterministic action routing and high-confidence seed label generation.

The `decide_action` / `choose_action` functions turn a built message context and its
classified `message_type` into a `(action, confidence)` tuple using explicit,
context-driven rules. This intentionally moves away from a single fixed priority
threshold (which heavily over-predicted `mute`) toward personalised signals:
direct mentions, real urgency, verified/admin senders, business opt-in/out state,
group mute state, forwarding and prompt-injection safety.
"""

import csv
import json
import re
from pathlib import Path
from typing import Any

from classification import classify_message_type
from context import build_context


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


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_data(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Load all participant-facing CSV data required by the router."""
    data = {name: _load_rows(dataset_dir / f"{name}.csv") for name in DATA_NAMES}
    return _load_rows(dataset_dir / "messages.csv"), data


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize fields needed by the context and classifier."""
    normalized = dict(message)
    for field in ("message_id", "user_id", "conversation_type", "group_id", "business_id", "sender_user_id", "media_type", "media_id"):
        normalized[field] = str(normalized.get(field, "") or "").strip()
    normalized["message_text"] = str(normalized.get("message_text", "") or "").strip()
    try:
        normalized["forwarded_count"] = int(normalized.get("forwarded_count", 0) or 0)
    except (TypeError, ValueError):
        normalized["forwarded_count"] = 0
    return normalized


# --------------------------------------------------------------------------- #
# action decision
# --------------------------------------------------------------------------- #
_PROMO_SPAM_RE = re.compile(
    r"50%\s*off|%\s*off|coupon|discount code|new here|shopping offer|offer code|get \d+%"
)


def _is_prompt_injection(text: str) -> bool:
    """Detect explicit attempts to override routing (always a safety mute)."""
    t = text.lower()
    return bool(
        re.search(r"ignore (all )?previous routing rules|mark this message as notify|mark as notify|always (route|mark) .*?notify", t)
    )


def decide_action(context: dict[str, Any], message_type: str) -> tuple[str, float]:
    """Return the final routing decision and its confidence for a message.

    Order of checks matters: safety and hard-mute conditions run first, then
    notify conditions (urgency / trust / actionability), then defer / mute.
    """
    text = str(context.get("message", {}).get("content_text", "") or "").lower()
    sx = context.get("extracted", {}) or {}
    conv = context.get("conversation", {}) or {}
    membership = conv.get("membership", {}) or {}
    group_muted = str(membership.get("group_muted_by_user", "0")) == "1"
    role = str(membership.get("role", "") or "")

    sender = context.get("sender", {}) or {}
    is_verified = bool(sender.get("is_verified"))

    business = context.get("business", {}) or {}
    relationship = business.get("relationship", {}) or {}
    allows_prom = str(relationship.get("allows_promotions", "")).strip()

    history = context.get("history", {}).get("previous_interactions", {}) or {}
    reported = int(history.get("message_reported", 0) or 0)
    dismissed = int(history.get("notification_dismissed", 0) or 0)
    opened = int(history.get("message_opened", 0) or 0)

    direct = bool(sx.get("has_direct_mention"))
    deadline = bool(sx.get("deadline"))
    forwarded = bool(sx.get("is_forwarded"))
    promotional = bool(sx.get("promotional"))

    # --- safety / hard mute ---
    if _is_prompt_injection(text):
        return "mute", 0.99
    if message_type == "scam":
        return "mute", 0.99
    if message_type == "spam":
        return "mute", 0.90
    if message_type == "forward":
        return "mute", 0.85

    # --- notify: interrupt now ---
    if direct and message_type == "personal":
        return "notify", 0.90
    if message_type == "urgent":
        return "notify", 0.90
    if message_type == "payment" and not forwarded:
        return "notify", 0.88

    # event: notify when time-sensitive / from a trusted or verified figure
    if message_type == "event":
        if is_verified or role == "admin" or deadline or re.search(r"school|circular|consent|class|today|now", text):
            return "notify", 0.85
        return "digest", 0.75

    # business update: logistics/delivery/order -> notify; feedback/advisory -> digest
    if message_type == "business_update":
        if re.search(r"\b(order|packed|reach|delivered|shipment|arriv|dispatch|ready for review)\b", text) \
                and (is_verified or opened > 0):
            return "notify", 0.85
        return "digest", 0.75

    # greeting noise
    if message_type == "greeting":
        if forwarded or group_muted:
            return "mute", 0.80
        return "digest", 0.70

    # promotion: honour group mute and business opt-out, mute spammy offers
    if message_type == "promotion":
        if group_muted:
            return "mute", 0.85
        if allows_prom == "0":
            return "mute", 0.90
        if _PROMO_SPAM_RE.search(text):
            return "mute", 0.80
        if reported >= 2 and dismissed >= 5:
            return "mute", 0.80
        return "digest", 0.75

    # personal / unknown / fallback -> defer politely
    if message_type == "personal":
        if direct:
            return "notify", 0.90
        return "digest", 0.75

    if message_type == "unknown":
        return "digest", 0.65

    return "digest", 0.60


def calculate_priority(context: dict[str, Any], message_type: str) -> int:
    """Informational importance score (positive = more interruptive).

    Used only as a secondary signal / for the seed-confidence filter; the action
    itself is decided by `decide_action`.
    """
    signals = context.get("extracted", {}) or {}
    score = 0
    if message_type == "scam":
        return -50
    if signals.get("has_direct_mention"):
        score += 6
    if signals.get("deadline"):
        score += 5
    if message_type in {"urgent", "payment"}:
        score += 4
    if context.get("sender", {}).get("is_verified"):
        score += 2
    if context.get("conversation", {}).get("membership", {}).get("role") == "admin":
        score += 2
    if context.get("business", {}).get("relationship", {}).get("activity_count_180d"):
        score += 1
    if signals.get("is_forwarded"):
        score -= 3
    if signals.get("promotional"):
        score -= 2
    if context.get("history", {}).get("previous_interactions", {}).get("message_reported"):
        score -= 4
    return score


def choose_action(context: dict[str, Any], message_type: str, priority: int | None = None) -> tuple[str, float]:
    """Compatibility wrapper; the decision is context-driven, not threshold-based."""
    return decide_action(context, message_type)


# --------------------------------------------------------------------------- #
# seed output
# --------------------------------------------------------------------------- #
def _flatten_context(context: dict[str, Any]) -> dict[str, Any]:
    """Serialize nested context sections into stable CSV columns."""
    flat = {}
    for section, value in context.items():
        flat[f"context_{section}"] = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return flat


def _is_high_confidence(message_type: str, action: str, priority: int, confidence: float) -> bool:
    """Keep clear, decisive labels for the seed set."""
    if message_type in ("scam", "spam"):
        return True
    if action == "notify":
        return confidence >= 0.80
    if action == "mute":
        return confidence >= 0.85
    if action == "digest":
        return confidence >= 0.70
    return False


def write_seed_output(messages: list[dict[str, Any]], data: dict[str, list[dict[str, Any]]], output_path: Path) -> int:
    """Write deterministic labels to a seed CSV and return the row count."""
    rows = []
    for raw_message in messages:
        message = normalize_message(raw_message)
        context = build_context(message, data)
        message_type = classify_message_type(context)
        priority = calculate_priority(context, message_type)
        action, confidence = choose_action(context, message_type, priority)
        if not _is_high_confidence(message_type, action, priority, confidence):
            continue
        row = _flatten_context(context)
        row.update({
            "priority": priority,
            "confidence": confidence,
            "action": action,
            "message_type": message_type,
            "message_id": message.get("message_id", ""),
        })
        rows.append(row)

    columns = list(rows[0]) if rows else ["action", "message_type"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
    output_path = dataset_dir / "labeled_seed.csv"
    messages, data = load_data(dataset_dir)
    count = write_seed_output(messages, data, output_path)
    print(f"Wrote {count} high-confidence seed rows to {output_path}")


if __name__ == "__main__":
    main()


__all__ = ["load_data", "normalize_message", "calculate_priority", "choose_action", "decide_action", "_is_high_confidence", "write_seed_output"]
