from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from Config.model import get_model
from Agents.common import latest_user_query


MODEL = get_model()


class SplitTask(BaseModel):
    sub_query: str = Field(min_length=1)
    intent: Literal["POLICY_QUERY", "INCIDENT", "SERVICE_REQUEST", "GENERAL"]
    service_type: Literal["HR", "Finance", "IT", "Travel", "GENERAL"]


class SplitOutput(BaseModel):
    tasks: List[SplitTask] = Field(min_length=1)


SPLITTER_MODEL = MODEL.with_structured_output(SplitOutput)


def intent_splitter_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    user_query = latest_user_query(state)
    if not user_query:
        return {
            "tasks": [{"sub_query": "", "intent": "GENERAL", "service_type": "GENERAL", "status": "PENDING"}],
            "current_task_index": 0,
            "awaiting_confirmation": False,
            "confirmed": False,
            "user_query": "",
        }

    if state.get("status") == "AWAITING_USER" and state.get("tasks"):
        return {"user_query": user_query}

    existing_tasks = state.get("tasks", [])
    if existing_tasks and state.get("current_task_index", 0) < len(existing_tasks):
        return {"user_query": user_query}

    prompt = [
        SystemMessage(
            content=(
                "Split user request into actionable tasks.\n"
                "Return one or more tasks with intent and service_type.\n"
                "Allowed intent: POLICY_QUERY, INCIDENT, SERVICE_REQUEST, GENERAL.\n"
                "Allowed service_type: HR, Finance, IT, Travel, GENERAL.\n"
                "If request is single intent, return one task.\n"
            )
        ),
        HumanMessage(content=user_query),
    ]

    try:
        output = SPLITTER_MODEL.invoke(prompt, config=config)
        tasks = [
            {
                "sub_query": t.sub_query.strip(),
                "intent": t.intent,
                "service_type": t.service_type,
                "status": "PENDING",
            }
            for t in output.tasks
            if t.sub_query.strip()
        ]
    except Exception:
        tasks = []

    if not tasks:
        tasks = [
            {
                "sub_query": user_query,
                "intent": "GENERAL",
                "service_type": "GENERAL",
                "status": "PENDING",
            }
        ]

    requires_ticket = any(t["intent"] in {"INCIDENT", "SERVICE_REQUEST"} for t in tasks)
    awaiting_confirmation = len(tasks) > 1 and requires_ticket

    return {
        "tasks": tasks,
        "current_task_index": 0,
        "awaiting_confirmation": awaiting_confirmation,
        "confirmed": False,
        "user_query": user_query,
        "status": "IN_PROGRESS",
        "error": None,
    }

