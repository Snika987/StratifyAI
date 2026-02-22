from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from RAG.retrieve import search_knowledge_base
from Agents.common import RAG_THRESHOLD


def policy_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    try:
        query = state.get("user_query", "")
        results = search_knowledge_base(query, top_k=3)
        if not results:
            return {
                "rag_context": "",
                "rag_score": 0.0,
                "rag_found": False,
                "action": "ANSWER",
                "status": "READY_FOR_DECISION",
            }

        top_score = max(float(r.get("score", 0.0)) for r in results)
        rag_context = "\n\n---\n\n".join(
            str(r.get("chunk_text", "")) for r in results if r.get("chunk_text")
        )

        return {
            "rag_context": rag_context,
            "rag_score": top_score,
            "rag_found": top_score >= RAG_THRESHOLD,
            "action": "ANSWER",
            "status": "READY_FOR_DECISION",
        }
    except Exception as e:
        return {
            "rag_context": "",
            "rag_score": 0.0,
            "rag_found": False,
            "action": "REJECT",
            "status": "FAILED",
            "error": f"policy_error: {e}",
        }
