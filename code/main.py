"""End-to-end entry point for the deterministic message router.

Pipeline (per the original architecture):

    load_data -> normalize_message -> build_context -> classify_message_type
    -> calculate_priority -> choose_action -> select_evidence -> write_output

Every incoming message in dataset/messages.csv gets exactly one prediction row
in dataset/output.csv with the contract columns:

    message_id,action,message_type,reason,confidence,evidence_message_ids

Usage (from the repository root):

    python code/main.py                      # route all messages -> dataset/output.csv
    python code/main.py --output out.csv     # write predictions to a custom path
    python code/main.py --limit 5            # smoke-test on the first N messages
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from classification import classify_message_type
from context import build_context
from seed_pipeline import (
    _is_prompt_injection,
    calculate_priority,
    choose_action,
    load_data,
    normalize_message,
)

OUTPUT_COLUMNS = (
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)

EVIDENCE_LIMIT = 3


def select_evidence(context: dict[str, Any], limit: int = EVIDENCE_LIMIT) -> str:
    """Pick the most relevant historical message ids for this decision.

    ``context["history"]["similar_messages"]`` already holds the receiving
    user's past messages ranked by token overlap with the incoming text in a
    deterministic order, so the top ids are used as evidence. Returns "none"
    when no useful historical message exists.
    """
    similar = context.get("history", {}).get("similar_messages", []) or []
    ids: list[str] = []
    for row in similar:
        message_id = str(row.get("message_id", "") or "").strip()
        if message_id and message_id not in ids:
            ids.append(message_id)
        if len(ids) >= limit:
            break
    return ";".join(ids) if ids else "none"


def build_reason(context: dict[str, Any], message_type: str, action: str) -> str:
    """Build a short human-readable reason mirroring decide_action's checks."""
    low = str(context.get("message", {}).get("content_text", "") or "").lower()
    sx = context.get("extracted", {}) or {}
    membership = context.get("conversation", {}).get("membership", {}) or {}
    group_muted = str(membership.get("group_muted_by_user", "0")) == "1"
    role = str(membership.get("role", "") or "")
    is_verified = bool(context.get("sender", {}).get("is_verified"))
    relationship = context.get("business", {}).get("relationship", {}) or {}
    allows_promotions = str(relationship.get("allows_promotions", "")).strip()
    interactions = context.get("history", {}).get("previous_interactions", {}) or {}
    reported = int(interactions.get("message_reported", 0) or 0)
    dismissed = int(interactions.get("notification_dismissed", 0) or 0)

    # Safety mutes first, matching decide_action's ordering.
    if _is_prompt_injection(low):
        return "Message tries to override routing rules, so it was suppressed as unsafe."
    if message_type == "scam":
        if sx.get("otp"):
            return "Message asks for an OTP or login code after prompting a reply, which looks like a scam."
        return "Message carries a suspicious link with claim-or-verify language, so it was muted as a scam."
    if message_type == "spam":
        if not low:
            return "Empty message from a sender the user opted out of, so it was muted."
        return "Chain-forward or prize-claim language typical of spam, so it was muted."
    if message_type == "forward":
        return "Widely forwarded content with no personal relevance, so it was muted."

    if action == "notify":
        if message_type == "urgent":
            if sx.get("has_direct_mention"):
                return "Direct mention of the user with a real deadline that needs attention now."
            return "Time-sensitive request that should interrupt the user."
        if message_type == "payment":
            return "Payment message from a known relationship that should be acted on now."
        if message_type == "event":
            if is_verified:
                return "Time-sensitive event update from a verified sender that the user needs now."
            if role == "admin":
                return "Time-sensitive event update from a group admin that the user needs now."
            return "Same-day event update with a deadline that the user needs now."
        if message_type == "business_update":
            return "Delivery or order update from a verified business that matches recent activity."
        if message_type == "personal":
            return "Direct mention of the user in a personal conversation."

    if action == "mute":
        if message_type == "greeting":
            if sx.get("is_forwarded"):
                return "Forwarded greeting with no personal value, so it was muted."
            return "Routine greeting inside a group the user has muted."
        if message_type == "promotion":
            if group_muted:
                return "Promotional post inside a group the user has muted."
            if allows_promotions == "0":
                return "Promotional message from a business the user opted out of."
            if reported >= 2 and dismissed >= 5:
                return "Promotional content the user repeatedly reported and dismissed."
            return "Low-value repetitive promotional offer."

    # Digest fallbacks by message type.
    if message_type == "event":
        return "Event update without immediate time pressure, safe to batch for later."
    if message_type == "business_update":
        return "Non-critical business update, safe to batch for later."
    if message_type == "greeting":
        return "Routine greeting that can wait for later."
    if message_type == "promotion":
        return "Promotional content from a followed sender, safe to batch for later."
    if message_type == "payment":
        return "Forwarded payment content without provenance, batched for later review."
    if message_type == "personal":
        return "Ordinary personal chat with no direct mention or urgency."
    return "Could not determine a strong intent, so it was deferred to the digest."


def route_message(message: dict[str, Any], data: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    """Run the full decision pipeline for one normalized incoming message."""
    context = build_context(message, data)
    message_type = classify_message_type(context)
    priority = calculate_priority(context, message_type)
    action, confidence = choose_action(context, message_type, priority)
    return {
        "message_id": str(message.get("message_id", "") or ""),
        "action": action,
        "message_type": message_type,
        "reason": build_reason(context, message_type, action),
        "confidence": f"{float(confidence):.2f}",
        "evidence_message_ids": select_evidence(context),
    }


def write_output(
    messages: list[dict[str, Any]],
    data: dict[str, list[dict[str, Any]]],
    output_path: Path,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Route every incoming message and write the contract-compliant CSV."""
    selected = messages if limit is None else messages[: max(limit, 0)]
    rows = [route_message(normalize_message(raw), data) for raw in selected]
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_pipeline(dataset_dir: Path, output_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    """Load the dataset, route every message, write predictions, print a summary."""
    messages, data = load_data(Path(dataset_dir))
    rows = write_output(messages, data, Path(output_path), limit=limit)
    counts = Counter(row["action"] for row in rows)
    summary = " / ".join(f"{action}={counts.get(action, 0)}" for action in ("notify", "digest", "mute"))
    print(f"Routed {len(rows)} of {len(messages)} messages -> {output_path}")
    print(f"Action distribution: {summary}")
    return rows


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Route dataset/messages.csv into an output.csv.")
    parser.add_argument("--dataset-dir", type=Path, default=root / "dataset")
    parser.add_argument("--output", type=Path, default=root / "dataset" / "output.csv")
    parser.add_argument("--limit", type=int, default=None, help="Route only the first N messages.")
    args = parser.parse_args()
    run_pipeline(args.dataset_dir, args.output, limit=args.limit)


if __name__ == "__main__":
    main()


__all__ = [
    "build_context",
    "classify_message_type",
    "calculate_priority",
    "choose_action",
    "load_data",
    "normalize_message",
    "build_reason",
    "route_message",
    "select_evidence",
    "write_output",
    "run_pipeline",
]

