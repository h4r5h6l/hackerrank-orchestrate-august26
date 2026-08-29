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
		(not text and str(relationship.get("allows_promotions", "1")) == "0")
		or re.search(r"send this to \d+ people|forward this|you have won|claim your prize", text)
	)


def is_business_update(context: dict[str, Any]) -> bool:
	text = str(context.get("message", {}).get("content_text", "")).lower()
	return bool(re.search(
		r"\b(?:delivery|delivered|shipment|order update|account update|service update|ready for review|feedback|safety advisory)\b",
		text,
	))


def is_event(context: dict[str, Any]) -> bool:
	text = str(context.get("message", {}).get("content_text", "")).lower()
	return bool(re.search(
		r"\b(?:event|meeting|school|bus|class|circular|form|registration|schedule|concert|webinar|deadline|appointment|timing|consent)\b",
		text,
	))


def _has_payment_intent(context: dict[str, Any]) -> bool:
	text = str(context.get("message", {}).get("content_text", "")).lower()
	if re.search(r"failed[- ]payment\s+screenshots?", text):
		return False
	if re.search(r"(?:never|do not|don't|will never) ask for .*payment", text):
		return False
	return bool(re.search(
		r"\b(?:invoice|receipt|amount due|payment reminder|refund|transfer|paid|pay|upi|fee|charged)\b",
		text,
	))


def _has_real_urgency(context: dict[str, Any]) -> bool:
	text = str(context.get("message", {}).get("content_text", "")).lower()
	if re.search(r"nothing urgent|no urgency|no rush|not urgent", text):
		return False
	return _has_urgency_signal(context.get("extracted", {}))


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

	if signals.get("has_direct_mention") and _has_real_urgency(context):
		return "urgent"

	if _has_payment_intent(context):
		return "payment"

	if signals.get("promotional"):
		return "promotion"

	if signals.get("is_greeting"):
		return "greeting"
	if _has_real_urgency(context) and not is_event(context) and (
		signals.get("asks_for_reply")
		or re.search(r"come online|quick help|mins? max|last[- ]minute|escalation starts", text)
	):
		return "urgent"
	if is_event(context) or _has_real_urgency(context):
		return "event"
	if not text and context.get("conversation", {}).get("recent_messages"):
		sender_id = context.get("message", {}).get("sender_id")
		if any(row.get("sender_user_id") == sender_id for row in context["conversation"]["recent_messages"]):
			return "personal"
	if is_business_update(context):
		return "business_update"
	if re.search(r"\b(?:good morning|good evening|good night|hello everyone|hi everyone)\b", text):
		return "greeting"
	if signals.get("is_forwarded"):
		return "forward"
	if signals.get("asks_for_reply"):
		return "personal"
	if context.get("conversation", {}).get("chat_type") == "personal" and any(
		row.get("sender_user_id") == context.get("message", {}).get("sender_id")
		for row in context.get("history", {}).get("similar_messages", [])
	):
		return "personal"
	if context.get("conversation", {}).get("chat_type") == "group" and text:
		return "personal"
	return "unknown"


__all__ = ["classify_message_type"]