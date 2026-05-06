"""Research-workflow-specific chat plumbing on top of `chat_lib`.

`tools.py` -- Pydantic-typed tool handlers per phase. Read tools (search,
get_detail) hit dealcloud directly; mutating tools (add/remove/clear,
advance_phase) open a transaction, lock the session row, append a new
session_version, and emit a `version_created` side event the
orchestrator forwards to its SSE stream.

`orchestrator.py` -- glues `chat_lib.run_chat_turn` to the research
schema: loads chat history from session_chat_message, persists new
user/assistant/tool messages with the right pre/post_version_id and
parent_message_id links, and translates loop events into the SSE event
shape the frontend consumes.

The split mirrors `chat_lib`'s contract: `tools.py` provides handlers;
`orchestrator.py` provides ctx/on_event. Both are research-specific;
neither leaks into chat_lib.
"""
