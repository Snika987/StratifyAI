from typing import Any, Dict


def task_progress_node(state: Dict[str, Any], config) -> Dict[str, Any]:
    tasks = list(state.get("tasks", []))
    idx = int(state.get("current_task_index", 0))

    if not tasks:
        return {"status": "COMPLETED"}

    if 0 <= idx < len(tasks):
        tasks[idx]["status"] = "COMPLETED"

    next_idx = idx + 1
    if next_idx >= len(tasks):
        return {"tasks": tasks, "status": "COMPLETED"}

    return {
        "tasks": tasks,
        "current_task_index": next_idx,
        "status": "IN_PROGRESS",
        # reset per-task execution outputs only
        "collected_fields": {},
        "ticket_id": None,
        "ticket_payload": {},
        "rag_context": "",
        "rag_score": 0.0,
        "rag_found": False,
        "required_fields": [],
        "missing_fields": [],
        "validation_passed": False,
        "action": "NONE",
        "error": None,
    }


def route_after_task_progress(state: Dict[str, Any]) -> str:
    if state.get("status") == "COMPLETED":
        return "end"
    return "router"
