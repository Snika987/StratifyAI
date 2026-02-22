from typing import Any, Dict

from langchain_core.runnables import RunnableConfig

from Agents.common import normalize_yes_no


def confirmation_handler_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    if not state.get("awaiting_confirmation", False):
        return {}

    user_text = str(state.get("user_query", "")).strip().lower()
    decision = normalize_yes_no(user_text)

    if decision is True:
        return {
            "confirmed": True,
            "awaiting_confirmation": False,
            "confirmation_prompt": None,
            "action": "NONE",
            "status": "IN_PROGRESS",
            "error": None,
        }

    if decision is False:
        return {
            "confirmed": False,
            "awaiting_confirmation": False,
            "confirmation_prompt": None,
            "tasks": [],
            "action": "ANSWER",
            "status": "COMPLETED",
            "error": "user_declined_multi_ticket_confirmation",
        }

    return {
        "action": "ASK_USER",
        "status": "AWAITING_USER",
    }


def route_after_confirmation_handler(state: Dict[str, Any]) -> str:
    tasks = state.get("tasks", []) or []
    idx = state.get("current_task_index", 0)
    has_pending_tasks = bool(tasks) and idx < len(tasks)

    # Invalid/ambiguous confirmation reply; ask again.
    if state.get("awaiting_confirmation", False) and state.get("status") == "AWAITING_USER":
        return "response"

    # User declined confirmation, tasks cleared.
    if not tasks and state.get("action") == "ANSWER" and state.get("status") == "COMPLETED":
        return "response"

    # User confirmed and there is an active task queue.
    if state.get("confirmed", False) and has_pending_tasks:
        return "router"

    # Normal flow.
    return "intent_splitter"
