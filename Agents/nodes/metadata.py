from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from Agents.common import (
    ALLOWED_DEPARTMENTS,
    detect_department,
    detect_priority,
    required_fields_for_intent,
)


def metadata_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    try:
        user_query = state.get("user_query", "")
        intent = state.get("intent", "GENERAL")
        service_type = state.get("service_type", "GENERAL")

        required = required_fields_for_intent(intent, service_type)
        collected = dict(state.get("collected_fields", {}))

        if not collected.get("description"):
            collected["description"] = user_query.strip()

        if not collected.get("priority"):
            collected["priority"] = detect_priority(user_query)

        if not collected.get("department"):
            dept = detect_department(user_query)
            if dept:
                collected["department"] = dept
            elif service_type in ALLOWED_DEPARTMENTS:
                collected["department"] = service_type

        if "amount_context" in required and not collected.get("amount_context"):
            if any(tok in user_query.lower() for tok in ["$", "usd", "amount", "invoice", "reimburse"]):
                collected["amount_context"] = "present"

        missing = [f for f in required if not collected.get(f)]

        return {
            "required_fields": required,
            "collected_fields": collected,
            "missing_fields": missing,
            "status": "IN_PROGRESS",
            "error": None,
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "action": "REJECT",
            "error": f"metadata_error: {e}",
        }
