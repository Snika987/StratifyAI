from typing import Any, Dict, Optional
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langchain_core.messages import HumanMessage

from Database.checkpointer import get_checkpointer
from State.state import AgentState
from Agents.nodes.confirmation_handler import (
    confirmation_handler_node,
    route_after_confirmation_handler,
)
from Agents.nodes.intent_splitter import intent_splitter_node
from Agents.nodes.confirmation import confirmation_node, route_after_confirmation
from Agents.nodes.router import router_node, route_after_router
from Agents.nodes.policy import policy_node
from Agents.nodes.metadata import metadata_node
from Agents.nodes.validation import validation_node
from Agents.nodes.decision import decision_node, route_after_decision
from Agents.nodes.execution import execution_node
from Agents.nodes.response import response_node
from Agents.nodes.task_progress import task_progress_node, route_after_task_progress


def route_after_response(state: Dict[str, Any]) -> str:
    # Wait for user reply in slot-filling/confirmation paths.
    if state.get("status") == "AWAITING_USER":
        return "end"
    # User explicitly declined multi-ticket confirmation and tasks were cleared.
    if not state.get("tasks") and state.get("action") == "ANSWER" and state.get("status") == "COMPLETED":
        return "end"
    # Explicit rejection stops processing.
    if state.get("action") == "REJECT":
        return "end"
    return "task_progress"


def build_agent():
    builder = StateGraph(AgentState)

    builder.add_node("confirmation_handler", confirmation_handler_node)
    builder.add_node("intent_splitter", intent_splitter_node)
    builder.add_node("confirmation", confirmation_node)
    builder.add_node("router", router_node)
    builder.add_node("policy", policy_node)
    builder.add_node("metadata", metadata_node)
    builder.add_node("validation", validation_node)
    builder.add_node("decision", decision_node)
    builder.add_node("execution", execution_node)
    builder.add_node("response", response_node)
    builder.add_node("task_progress", task_progress_node)

    builder.add_edge(START, "confirmation_handler")
    builder.add_conditional_edges(
        "confirmation_handler",
        route_after_confirmation_handler,
        {"intent_splitter": "intent_splitter", "router": "router", "response": "response"},
    )
    builder.add_edge("intent_splitter", "confirmation")
    builder.add_conditional_edges(
        "confirmation",
        route_after_confirmation,
        {"router": "router", "response": "response"},
    )
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {"policy": "policy", "metadata": "metadata", "response": "response"},
    )
    builder.add_edge("policy", "decision")
    builder.add_edge("metadata", "validation")
    builder.add_edge("validation", "decision")
    builder.add_conditional_edges(
        "decision",
        route_after_decision,
        {"execution": "execution", "response": "response"},
    )
    builder.add_edge("execution", "response")
    builder.add_conditional_edges(
        "response",
        route_after_response,
        {"task_progress": "task_progress", "end": END},
    )
    builder.add_conditional_edges(
        "task_progress",
        route_after_task_progress,
        {"router": "router", "end": END},
    )

    checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)


agent_graph = build_agent()


def invoke_agent(user_query: str, thread_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    tid = thread_id or f"thread_{uuid4()}"
    uid = user_id or "anonymous"

    # Only provide turn input so checkpointed state remains intact across turns.
    turn_state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
    }

    config = {"configurable": {"thread_id": tid, "user_id": uid}}
    # Drain stream fully so all checkpoints are written for this turn.
    for _ in agent_graph.stream(turn_state, config=config):
        pass

    snapshot = agent_graph.get_state(config)
    return dict(snapshot.values) if snapshot and snapshot.values else {}
