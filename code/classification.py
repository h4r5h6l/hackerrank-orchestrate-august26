"""Deterministic message-type classification rules."""

import re
from typing import Any


def _has_urgency_signal(signals: dict[str, Any]) -> bool:
	return bool(signals.get("has_urgency_signal", signals.get("deadline")))


def is_spam(context: dict[str, Any]) -> bool:
	signals = context.get("extracted", {})
	relationship = context.get("business", {}).get("relationship", {})
	text = str(context.get("message", {}).get("content_text", "")).lower()
	return bool(
		(signals.get("promotional") and (
			str(relationship.get("allows_promotions", "1")) == "0"
		))
		or re.search(r"send this to \d+ people|forward this|you have won|claim your prize", text)
	)


def is_business_update(context: dict[str, Any]) -> bool:
	text = str(context.get("message", {}).get("content_text", "")).lower()
	return bool(re.search(
		r"\b(?:delivery|delivered|shipment|order update|booking|appointment|account update|expire|renewal|pickup|service update|ready for review|scheduled)\b",
		text,
	))


def is_event(context: dict[str, Any]) -> bool:
	text = str(context.get("message", {}).get("content_text", "")).lower()
	return bool(re.search(
		r"\b(?:event|meeting|school|bus|class|circular|form|registration|schedule|concert|webinar|deadline|tomorrow|today)\b",
		text,
	))


def classify_message_type(context: dict[str, Any]) -> str:
	"""Return one allowed message_type value for a built message context."""
	signals = context.get("extracted", {})
	text = str(context.get("message", {}).get("content_text", "")).lower()
	is_otp_disclaimer = bool(re.search(r"never ask for|do not ask for|don't ask for|will never ask", text))

	if signals.get("suspicious_link") or (
		signals.get("otp") and (signals.get("asks_for_reply") or signals.get("payment"))
		and not is_otp_disclaimer
	):
		return "scam"

	if is_spam(context):
		return "spam"

	if signals.get("payment") or re.search(
		r"\b(?:invoice|receipt|amount due|payment reminder|refund|transfer|paid|pay)\b",
		text,
	):
		return "payment"

	if signals.get("promotional"):
		return "promotion"

	if is_business_update(context):
		return "business_update"

	if signals.get("has_direct_mention") and _has_urgency_signal(signals):
		return "urgent"
	if is_event(context) or _has_urgency_signal(signals):
		return "event"
	if signals.get("is_greeting"):
		return "greeting"
	if signals.get("is_forwarded"):
		return "forward"
	if signals.get("asks_for_reply"):
		return "personal"
	return "unknown"


__all__ = ["classify_message_type"]