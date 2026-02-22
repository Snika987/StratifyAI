from typing import Any, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from Config.model import get_model
from State.state import IntentType, ServiceType
from Agents.common import get_current_task, latest_user_query

MODEL = get_model()


class RouterOutput(BaseModel):
    intent: IntentType
    service_type: ServiceType
    confidence: float = Field(ge=0.0, le=1.0)


ROUTER_MODEL = MODEL.with_structured_output(RouterOutput)


def router_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    try:
        if state.get("awaiting_confirmation", False):
            return {}

        user_query = latest_user_query(state)
        if not user_query:
            return {
                "intent": "GENERAL",
                "service_type": "GENERAL",
                "action": "ANSWER",
                "status": "READY_FOR_DECISION",
                "user_query": "",
            }

        if state.get("confirmed", False) and state.get("tasks"):
            current_task = get_current_task(state)
            if current_task:
                return {
                    "user_query": user_query,
                    "intent": current_task["intent"],
                    "service_type": current_task["service_type"],
                    "status": "IN_PROGRESS",
                    "error": None,
                }

        # Multi-turn continuation path: keep existing task classification and continue slot filling.
        if state.get("status") == "AWAITING_USER" and state.get("intent"):
            current_task = get_current_task(state)
            if current_task:
                return {
                    "user_query": user_query,
                    "intent": current_task["intent"],
                    "service_type": current_task["service_type"],
                    "status": "IN_PROGRESS",
                    "error": None,
                }
            return {
                "user_query": user_query,
                "intent": state.get("intent", "GENERAL"),
                "service_type": state.get("service_type", "GENERAL"),
                "status": "IN_PROGRESS",
                "error": None,
            }

        # When tasks exist, route according to current task without reclassification.
        current_task = get_current_task(state)
        if current_task:
            return {
                "user_query": user_query,
                "intent": current_task["intent"],
                "service_type": current_task["service_type"],
                "status": "IN_PROGRESS",
                "error": None,
            }

        prompt = [
            SystemMessage(
                content=(
                    "Classify request.\n"
                    "intent: POLICY_QUERY | INCIDENT | SERVICE_REQUEST | GENERAL\n"
                    "service_type: HR | Finance | IT | Travel | GENERAL\n"
                )
            ),
            HumanMessage(content=user_query),
        ]
        routed = ROUTER_MODEL.invoke(prompt, config=config)

        if routed.confidence < 0.55:
            return {
                "user_query": user_query,
                "intent": "GENERAL",
                "service_type": "GENERAL",
                "action": "ANSWER",
                "status": "READY_FOR_DECISION",
            }

        return {
            "user_query": user_query,
            "intent": routed.intent,
            "service_type": routed.service_type,
            "action": "NONE",
            "status": "IN_PROGRESS",
            "error": None,
        }
    except Exception as e:
        return {
            "intent": "GENERAL",
            "service_type": "GENERAL",
            "action": "REJECT",
            "status": "FAILED",
            "error": f"router_error: {e}",
        }


def route_after_router(state: Dict[str, Any]) -> str:
    if state.get("error"):
        return "response"
    if state.get("intent") == "POLICY_QUERY":
        return "policy"
    if state.get("intent") in ("INCIDENT", "SERVICE_REQUEST"):
        return "metadata"
    return "response"
