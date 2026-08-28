"""Deterministic message-type classification rules."""

import re
from typing import Any


def classify_message_type(context: dict[str, Any]) -> str:
	"""Return one allowed message_type value for a built message context."""
	extracted = context.get("extracted", {})
	text = str(context.get("message", {}).get("content_text", "")).lower()

	# Safety takes precedence over the apparent business intent.
	if extracted.get("suspicious_link") or (
		extracted.get("otp") and (extracted.get("asks_for_reply") or extracted.get("payment"))
	):
		return "scam"

	if extracted.get("payment") or re.search(
		r"\b(?:invoice|receipt|amount due|payment reminder|refund|transfer|paid|pay)\b",
		text,
	):
		return "payment"

	if re.search(
		r"\b(?:delivery|delivered|shipment|order update|booking|appointment|account update|expire|renewal|pickup|schedule|service update)\b",
		text,
	):
		return "business_update"

	if extracted.get("has_direct_mention") and extracted.get("deadline"):
		return "urgent"
	if extracted.get("deadline"):
		return "event"
	if extracted.get("promotional"):
		return "promotion"
	if extracted.get("is_greeting"):
		return "greeting"
	if extracted.get("is_forwarded"):
		return "forward"
	if extracted.get("asks_for_reply"):
		return "personal"
	return "unknown"


__all__ = ["classify_message_type"]