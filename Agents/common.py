from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage
from State.state import IntentType, ServiceType, Task

RAG_THRESHOLD = 0.75
ALLOWED_DEPARTMENTS = {"HR", "Finance", "IT", "Travel"}
ALLOWED_PRIORITIES = {"low", "medium", "high", "urgent"}


def latest_user_query(state: Dict[str, Any]) -> str:
    if state.get("user_query"):
        return state["user_query"]
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def normalize_yes_no(text: str) -> Optional[bool]:
    t = text.strip().lower()
    if t in {"yes", "y", "yeah", "yep", "confirm", "proceed"}:
        return True
    if t in {"no", "n", "nope", "cancel", "stop"}:
        return False
    return None


def get_current_task(state: Dict[str, Any]) -> Optional[Task]:
    tasks = state.get("tasks", [])
    idx = state.get("current_task_index", 0)
    if not tasks or idx < 0 or idx >= len(tasks):
        return None
    return tasks[idx]


def is_last_task(state: Dict[str, Any]) -> bool:
    tasks = state.get("tasks", [])
    idx = state.get("current_task_index", 0)
    return bool(tasks) and idx >= len(tasks) - 1


def detect_department(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in ["human resources", "leave", "payroll", "hr"]):
        return "HR"
    if any(k in t for k in ["reimbursement", "invoice", "expense", "finance", "payment"]):
        return "Finance"
    if any(k in t for k in ["laptop", "vpn", "access", "vm", "it", "system", "login"]):
        return "IT"
    if any(k in t for k in ["travel", "flight", "hotel", "trip", "booking"]):
        return "Travel"
    return None


def detect_priority(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["critical", "sev1", "urgent", "immediately", "asap", "outage"]):
        return "urgent"
    if any(k in t for k in ["high", "major", "blocked"]):
        return "high"
    if any(k in t for k in ["low", "minor", "whenever"]):
        return "low"
    return "medium"


def required_fields_for_intent(intent: IntentType, service_type: ServiceType) -> List[str]:
    if intent in ("INCIDENT", "SERVICE_REQUEST"):
        fields = ["description", "priority", "department"]
        if service_type == "Finance":
            fields.append("amount_context")
        return fields
    return []
