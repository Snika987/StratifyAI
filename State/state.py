from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

IntentType = Literal["POLICY_QUERY", "INCIDENT", "SERVICE_REQUEST", "GENERAL"]
ServiceType = Literal["HR", "Finance", "IT", "Travel", "GENERAL"]
ActionType = Literal["ANSWER", "ASK_USER", "CREATE_TICKET", "REJECT", "NONE"]
StatusType = Literal[
    "IN_PROGRESS",
    "READY_FOR_DECISION",
    "AWAITING_USER",
    "VALIDATED",
    "EXECUTED",
    "COMPLETED",
    "FAILED",
]


TaskStatus = Literal["PENDING", "COMPLETED"]


class Task(TypedDict):
    sub_query: str
    intent: IntentType
    service_type: ServiceType
    status: TaskStatus


class AgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    user_query: str
    intent: IntentType
    service_type: ServiceType
    rag_context: str
    rag_score: float
    rag_found: bool
    required_fields: List[str]
    collected_fields: Dict[str, Any]
    missing_fields: List[str]
    validation_passed: bool
    action: ActionType
    ticket_payload: Dict[str, Any]
    ticket_id: Optional[str]
    task_ticket_ids: List[str]
    status: StatusType
    error: Optional[str]
    tasks: List[Task]
    current_task_index: int
    awaiting_confirmation: bool
    confirmed: bool
    confirmation_prompt: Optional[str]
