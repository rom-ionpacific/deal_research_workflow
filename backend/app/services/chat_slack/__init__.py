"""Slack-side conversational engine for Todd. Wraps `chat_lib` with a
Slack-specific tool set, conversation persistence, and a Slack-friendly
event renderer (tool calls -> small context blocks; assistant text ->
section blocks)."""
from .orchestrator import run_slack_chat_turn

__all__ = ["run_slack_chat_turn"]
