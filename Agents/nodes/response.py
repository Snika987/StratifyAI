from typing import Any, Dict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from Config.model import get_model

MODEL = get_model()


def response_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    try:
        action = state.get("action", "NONE")
        intent = state.get("intent", "GENERAL")
        rag_found = state.get("rag_found", False)
        rag_context = state.get("rag_context", "")
        user_query = state.get("user_query", "")
        error = state.get("error")

        if state.get("confirmation_prompt") and state.get("status") == "AWAITING_USER":
            text = state["confirmation_prompt"]

        elif error == "user_declined_multi_ticket_confirmation":
            text = "Understood. No tickets were created."

        elif action == "REJECT":
            if error == "user_declined_multi_ticket_confirmation":
                text = "Understood. I will not create separate tickets for these requests."
            else:
                text = f"Request rejected. Reason: {error or 'validation failed'}"

        elif action == "ASK_USER":
            missing = state.get("missing_fields", [])
            text = (
                "I need additional details before I can proceed.\n"
                f"Missing fields: {', '.join(missing)}"
                if missing
                else "I need additional details before I can proceed."
            )

        elif action == "CREATE_TICKET" and state.get("ticket_id"):
            text = (
                "Ticket Created\n"
                f"ID: {state.get('ticket_id')}\n"
                f"Department: {state.get('ticket_payload', {}).get('department', 'N/A')}\n"
                f"Priority: {state.get('ticket_payload', {}).get('priority', 'N/A')}\n"
                "Status: open"
            )

        elif action == "ANSWER" and intent == "POLICY_QUERY":
            if not rag_found or not rag_context:
                text = "I don't know"
            else:
                prompt = [
                    SystemMessage(
                        content=(
                            "You are an enterprise support assistant. "
                            "Answer strictly from policy context. "
                            "If insufficient context, respond exactly: I don't know"
                        )
                    ),
                    HumanMessage(
                        content=f"User query:\n{user_query}\n\nPolicy context:\n{rag_context}\n\nAnswer briefly."
                    ),
                ]
                llm_resp = MODEL.invoke(prompt, config=config)
                text = str(llm_resp.content).strip() if llm_resp else "I don't know"

        else:
            llm_resp = MODEL.invoke(
                [
                    SystemMessage(content="Provide a concise, professional enterprise support response."),
                    HumanMessage(content=user_query),
                ],
                config=config,
            )
            text = str(llm_resp.content).strip() if llm_resp else "I don't know"

        if state.get("status") == "AWAITING_USER":
            final_status = "AWAITING_USER"
        elif state.get("status") == "FAILED":
            final_status = "FAILED"
        else:
            final_status = "COMPLETED"

        return {"messages": [AIMessage(content=text)], "status": final_status}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Request failed: {e}")], "status": "FAILED", "error": f"response_error: {e}"}
