from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from Tools.tools import create_ticket


def execution_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    if state.get("action") != "CREATE_TICKET":
        return {"status": state.get("status", "IN_PROGRESS")}

    try:
        fields = state.get("collected_fields", {})
        payload = {
            "department": str(fields["department"]),
            "description": str(fields["description"]),
            "priority": str(fields.get("priority", "medium")).lower(),
        }

        result = create_ticket.invoke(payload, config=config)
        all_ids = list(state.get("task_ticket_ids", []))
        if result.get("ticket_id"):
            all_ids.append(result["ticket_id"])
        return {
            "ticket_payload": payload,
            "ticket_id": result.get("ticket_id"),
            "task_ticket_ids": all_ids,
            "status": "EXECUTED",
            "error": None,
        }
    except Exception as e:
        return {"status": "FAILED", "action": "REJECT", "error": f"execution_error: {e}"}
