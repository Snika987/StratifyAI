from typing import Any, Dict


def decision_node(state: Dict[str, Any], config) -> Dict[str, Any]:
    if state.get("error"):
        return {"action": "REJECT", "status": "FAILED"}

    intent = state.get("intent", "GENERAL")

    if intent in ("POLICY_QUERY", "GENERAL"):
        return {"action": "ANSWER", "status": "READY_FOR_DECISION"}

    if intent in ("INCIDENT", "SERVICE_REQUEST"):
        if state.get("missing_fields"):
            return {"action": "ASK_USER", "status": "AWAITING_USER"}
        if not state.get("validation_passed", False):
            return {"action": "REJECT", "status": "FAILED"}
        if state.get("awaiting_confirmation") and not state.get("confirmed"):
            return {"action": "ASK_USER", "status": "AWAITING_USER"}
        return {"action": "CREATE_TICKET", "status": "READY_FOR_DECISION"}

    return {"action": "REJECT", "status": "FAILED", "error": "unsupported_intent"}


def route_after_decision(state: Dict[str, Any]) -> str:
    if state.get("action") == "CREATE_TICKET":
        return "execution"
    return "response"
