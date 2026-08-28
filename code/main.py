"""Entry point for the deterministic message router."""

from context import build_context
from classification import classify_message_type

__all__ = ["build_context", "classify_message_type"]
