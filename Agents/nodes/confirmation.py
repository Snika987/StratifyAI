from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig

from Agents.common import normalize_yes_no


def confirmation_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    tasks = state.get("tasks", [])
    if not state.get("awaiting_confirmation", False):
        return {}

    # First pass: ask for explicit confirmation.
    if state.get("confirmed", False):
        return {"confirmed": True, "awaiting_confirmation": False}

    user_text = str(state.get("user_query", ""))
    decision = normalize_yes_no(user_text)
    if decision is True:
        return {"confirmed": True, "awaiting_confirmation": False, "status": "IN_PROGRESS"}
    if decision is False:
        return {
            "confirmed": False,
            "awaiting_confirmation": False,
            "action": "REJECT",
            "status": "COMPLETED",
            "error": "user_declined_multi_ticket_confirmation",
        }

    lines: List[str] = [
        "I detected multiple requests:",
        "",
    ]
    for idx, task in enumerate(tasks, start=1):
        lines.append(f"{idx}. [{task.get('intent')}] {task.get('sub_query')}")
    lines.append("")
    lines.append("Do you want me to proceed with creating separate tickets? (yes/no)")

    return {
        "action": "ASK_USER",
        "status": "AWAITING_USER",
        "error": None,
        "confirmation_prompt": "\n".join(lines),
    }


def route_after_confirmation(state: Dict[str, Any]) -> str:
    if state.get("status") == "AWAITING_USER":
        return "response"
    if state.get("action") == "REJECT":
        return "response"
    return "router"
