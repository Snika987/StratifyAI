from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from RAG.retrieve import search_knowledge_base
from Agents.common import ALLOWED_DEPARTMENTS, ALLOWED_PRIORITIES


def validation_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    try:
        intent = state.get("intent", "GENERAL")
        if intent not in ("INCIDENT", "SERVICE_REQUEST"):
            return {"validation_passed": True, "status": "VALIDATED"}

        if state.get("missing_fields"):
            return {"validation_passed": False, "status": "AWAITING_USER"}

        fields = state.get("collected_fields", {})
        department = str(fields.get("department", "")).strip()
        priority = str(fields.get("priority", "")).strip().lower()
        description = str(fields.get("description", "")).strip()

        if department not in ALLOWED_DEPARTMENTS:
            return {"validation_passed": False, "status": "FAILED", "error": f"invalid_department: {department}"}
        if priority not in ALLOWED_PRIORITIES:
            return {"validation_passed": False, "status": "FAILED", "error": f"invalid_priority: {priority}"}
        if not description:
            return {"validation_passed": False, "status": "FAILED", "error": "invalid_description"}

        if intent == "SERVICE_REQUEST":
            policy_results = search_knowledge_base(f"{department} request policy: {description}", top_k=1)
            if policy_results:
                best = float(policy_results[0].get("score", 0.0))
                ctx = str(policy_results[0].get("chunk_text", ""))
                merged = state.get("rag_context", "")
                merged = f"{merged}\n\n{ctx}".strip() if merged else ctx
                return {
                    "rag_context": merged,
                    "rag_score": max(state.get("rag_score", 0.0), best),
                    "validation_passed": True,
                    "status": "VALIDATED",
                }

        return {"validation_passed": True, "status": "VALIDATED"}
    except Exception as e:
        return {"validation_passed": False, "status": "FAILED", "error": f"validation_error: {e}"}
