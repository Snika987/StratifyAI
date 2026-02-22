from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage
from State.state import IntentType, ServiceType

RAG_THRESHOLD = 0.82
ALLOWED_DEPARTMENTS = {"HR", "Finance", "IT", "Travel"}
ALLOWED_PRIORITIES = {"low", "medium", "high", "urgent"}


def latest_user_query(state: Dict[str, Any]) -> str:
    if state.get("user_query"):
        return state["user_query"]
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


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
