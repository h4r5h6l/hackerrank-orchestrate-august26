"""Entry point for the deterministic message router."""

from context import build_context
from classification import classify_message_type
from seed_pipeline import calculate_priority, choose_action, load_data, normalize_message

__all__ = [
	"build_context",
	"classify_message_type",
	"calculate_priority",
	"choose_action",
	"load_data",
	"normalize_message",
]
