from typing import Any, Dict, Optional
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langchain_core.messages import HumanMessage

from Database.checkpointer import get_checkpointer
from State.state import AgentState
from Agents.nodes.router import router_node, route_after_router
from Agents.nodes.policy import policy_node
from Agents.nodes.metadata import metadata_node
from Agents.nodes.validation import validation_node
from Agents.nodes.decision import decision_node, route_after_decision
from Agents.nodes.execution import execution_node
from Agents.nodes.response import response_node


def build_agent():
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("policy", policy_node)
    builder.add_node("metadata", metadata_node)
    builder.add_node("validation", validation_node)
    builder.add_node("decision", decision_node)
    builder.add_node("execution", execution_node)
    builder.add_node("response", response_node)

    builder.add_edge(START, "router")
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
    builder.add_edge("response", END)

    return builder.compile(checkpointer=get_checkpointer())


agent_graph = build_agent()


def invoke_agent(user_query: str, thread_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    tid = thread_id or f"thread_{uuid4()}"
    uid = user_id or "anonymous"

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "intent": "GENERAL",
        "service_type": "GENERAL",
        "rag_context": "",
        "rag_score": 0.0,
        "rag_found": False,
        "required_fields": [],
        "collected_fields": {},
        "missing_fields": [],
        "validation_passed": False,
        "action": "NONE",
        "ticket_payload": {},
        "ticket_id": None,
        "status": "IN_PROGRESS",
        "error": None,
    }

    config = {"configurable": {"thread_id": tid, "user_id": uid}}
    return agent_graph.invoke(initial_state, config=config)
