"""Generic, integration-agnostic chat infrastructure for Anthropic tool use.

This package contains primitives that any caller (the deal-research-workflow
chat, Todd-the-Walrus' Phase 3 conversational mode, future agents) can
reuse. It does NOT know about session_versions, Slack, or any specific
persistence layer -- callers wire in their own context dict and event
callback.

Boundary:
  * tool_registry: declare tools with Pydantic input models; emit Anthropic
    tool schemas; dispatch by name with input validation.
  * loop: run one user-turn-and-response cycle, streaming text deltas and
    dispatching tool calls until the model stops using tools. Sync handlers
    run in asyncio.to_thread so this is safe inside a FastAPI endpoint.
"""
from .tool_registry import Tool, ToolRegistry, ToolResult
from .loop import run_chat_turn

__all__ = ["Tool", "ToolRegistry", "ToolResult", "run_chat_turn"]
