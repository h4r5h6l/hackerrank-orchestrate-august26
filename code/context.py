"""Context construction for deterministic message routing."""

from datetime import datetime
import re
from typing import Any


def _index(rows: list[dict[str, Any]], *keys: str) -> dict[Any, dict[str, Any]]:
	indexed = {}
	for row in rows:
		key = tuple(row.get(field, "") for field in keys)
		if all(key):
			indexed[key[0] if len(keys) == 1 else key] = row
	return indexed


def _parse_time(value: Any) -> datetime | None:
	if isinstance(value, datetime):
		return value
	if not value:
		return None
	try:
		return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
	except ValueError:
		return None


def _text(row: dict[str, Any]) -> str:
	return str(row.get("message_text", row.get("text", "")) or "")


def _tokens(value: str) -> set[str]:
	return set(re.findall(r"[a-z0-9]{3,}", value.lower()))


def _extract_signals(message: dict[str, Any], content_text: str) -> dict[str, Any]:
	text = content_text.lower()
	user_id = str(message.get("user_id", ""))
	urls = re.findall(r"https?://[^\s]+|\b[a-z0-9.-]+\.(?:com|in|net|org)\b", text)
	mentions = re.findall(r"@[a-z0-9_]+", text)
	return {
		"urls": urls,
		"mentions": mentions,
		"has_direct_mention": f"@{user_id.lower()}" in mentions,
		"otp": bool(re.search(r"\botp\b|one[- ]time password|login code|verification code", text)),
		"deadline": bool(re.search(r"today|tonight|urgent|asap|before \d|by \d|expires?|last[- ]minute", text)),
		"payment": bool(re.search(r"pay|payment|fee|transfer|refund|upi|invoice|charged", text)),
		"opt_out": bool(re.search(r"reply\s+stop|unsubscribe|opt[- ]out", text)),
		"asks_for_reply": bool(re.search(r"reply|respond|confirm|let me know|can you", text)),
		"suspicious_link": bool(urls) and bool(re.search(r"verify|claim|release|reactivate|blocked|prize", text)),
		"promotional": bool(re.search(r"sale|offer|discount|deal|limited time|book now|rs\.?\s*\d", text)),
		"is_forwarded": int(message.get("forwarded_count", 0) or 0) > 0,
		"is_greeting": bool(re.fullmatch(r"\s*(hi|hello|good morning|good evening|good night|hey)[!.\s]*", text)),
	}


def _interaction_summary(events: list[dict[str, Any]]) -> dict[str, int]:
	fields = (
		"message_opened",
		"message_replied",
		"notification_dismissed",
		"muted_after_message",
		"message_reported",
	)
	return {field: sum(int(event.get(field, 0) or 0) for event in events) for field in fields}


def _similar_messages(message: dict[str, Any], history: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
	target_tokens = _tokens(_text(message))
	scored = []
	for row in history:
		overlap = len(target_tokens & _tokens(_text(row)))
		if overlap:
			scored.append((overlap, row))
	scored.sort(key=lambda item: (-item[0], str(item[1].get("message_id", ""))))
	return [row for _, row in scored[:limit]]


def build_context(message: dict[str, Any], data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
	"""Build all deterministic inputs needed to classify one incoming message."""
	user_id = message.get("user_id", "")
	group_id = message.get("group_id", "")
	business_id = message.get("business_id", "")
	message_time = _parse_time(message.get("created_at", message.get("timestamp")))

	users = _index(data.get("users", []), "user_id")
	groups = _index(data.get("groups", []), "group_id")
	businesses = _index(data.get("business_accounts", []), "business_id")
	memberships = _index(data.get("group_members", []), "group_id", "user_id")
	business_history = _index(data.get("user_business_history", []), "user_id", "business_id")

	history = [row for row in data.get("message_history", []) if row.get("user_id") == user_id]
	if message_time:
		history = [row for row in history if not _parse_time(row.get("created_at")) or _parse_time(row.get("created_at")) < message_time]
	history.sort(key=lambda row: str(row.get("created_at", "")))
	conversation_history = [row for row in history if row.get("group_id", "") == group_id and row.get("business_id", "") == business_id][-20:]

	events = [row for row in data.get("message_events", []) if row.get("user_id") == user_id]
	daily_rows = [row for row in data.get("daily_notification_summary", []) if row.get("user_id") == user_id]
	date_value = str(message.get("created_at", ""))[:10]
	daily_load = next((row for row in daily_rows if row.get("date") == date_value), None)

	media_id = message.get("media_id", "")
	media_type = message.get("media_type", "")
	media_rows = data.get("images" if media_type == "image" else "voice_notes", [])
	media = next((row for row in media_rows if row.get("media_id") == media_id), None)
	content_text = _text(message)

	sender = users.get(message.get("sender_user_id"), {})
	business = businesses.get(business_id, {})
	relationship = business_history.get((user_id, business_id), {})

	return {
		"message": {
			"id": message.get("message_id", message.get("id")),
			"text": _text(message),
			"content_text": content_text,
			"timestamp": message.get("created_at", message.get("timestamp")),
			"sender_id": message.get("sender_user_id"),
			"chat_id": group_id or business_id or message.get("sender_user_id"),
			"media_type": media_type,
			"media": media,
		},
		"sender": {
			"type": message.get("conversation_type"),
			"name": sender.get("display_name", sender.get("name")),
			"is_business": bool(business_id),
			"is_verified": str(business.get("verified", "0")) == "1",
			"record": sender,
		},
		"conversation": {
			"chat_type": message.get("conversation_type"),
			"record": groups.get(group_id, {}),
			"membership": memberships.get((group_id, user_id), {}),
			"recent_messages": conversation_history,
		},
		"user": {
			"record": users.get(user_id, {}),
			"notification_preferences": users.get(user_id, {}).get("do_not_disturb_window"),
			"quiet_hours": users.get(user_id, {}).get("do_not_disturb_window"),
		},
		"business": {
			"record": business,
			"relationship": relationship,
			"purchase_history": relationship.get("activity_count_180d", 0),
			"subscription": relationship.get("why_user_knows_account"),
		},
		"history": {
			"previous_interactions": _interaction_summary(events),
			"similar_messages": _similar_messages(message, history),
		},
		"daily_load": daily_load or {},
		"extracted": _extract_signals(message, content_text),
	}


__all__ = ["build_context"]