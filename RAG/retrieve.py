import json
import math
from typing import Any, List, Optional

import numpy as np

from Database.session import get_connection
from Config.model import get_embeddings


# Temporarily lowered for debugging/validation runs.
RAG_SCORE_THRESHOLD = 0.5


def _to_vector(value: Any) -> Optional[np.ndarray]:
    """Convert JSON/DB vector payload into 1D float vector."""
    if value is None:
        return None

    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, (list, tuple)):
        return None

    try:
        vec = np.asarray(parsed, dtype=np.float32).reshape(-1)
    except (ValueError, TypeError):
        return None

    if vec.size == 0:
        return None

    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def search_knowledge_base(query: str, top_k: int = 3):
    embeddings = get_embeddings()
    query_raw = embeddings.embed_query(query)
    query_vector = _to_vector(query_raw)

    if query_vector is None:
        print("[RAG] Query embedding generation failed: empty/invalid vector")
        return []

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM rag_documents")
    total_chunks = int(cur.fetchone()[0] or 0)

    cur.execute("""
        SELECT department, chunk_text, embedding
        FROM rag_documents
    """)
    rows = cur.fetchall()

    print(f"[RAG] Query: {query}")
    print(f"[RAG] Query embedding length: {len(query_vector)}")
    print(f"[RAG] Embeddings count in DB: {total_chunks}")
    print(f"[RAG] Number of stored chunks retrieved: {len(rows)}")

    scored_results = []
    raw_scores: List[float] = []
    dimension_mismatches = 0

    for dept, chunk_text, embedding_json in rows:
        stored_embedding = _to_vector(embedding_json)
        if stored_embedding is None:
            continue

        if stored_embedding.shape[0] != query_vector.shape[0]:
            dimension_mismatches += 1
            continue

        score = cosine_similarity(query_vector, stored_embedding)
        if math.isnan(score) or math.isinf(score):
            continue

        raw_scores.append(score)
        scored_results.append(
            {
                "department": dept,
                "chunk_text": chunk_text,
                "score": score,
            }
        )

    cur.close()
    conn.close()

    print(f"[RAG] Dimension mismatches skipped: {dimension_mismatches}")
    if raw_scores:
        top_preview = sorted(raw_scores, reverse=True)[: min(10, len(raw_scores))]
        print(f"[RAG] Similarity scores before threshold filtering (top): {[round(s, 4) for s in top_preview]}")
    else:
        print("[RAG] Similarity scores before threshold filtering: []")

    # Sort by similarity descending
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = scored_results[:top_k]

    if not top_results:
        print("[RAG] No candidate chunks after scoring")
        return []

    print(f"[RAG] Top score before threshold: {top_results[0]['score']:.4f}")
    if top_results[0]["score"] < RAG_SCORE_THRESHOLD:
        print(f"[RAG] Top score below threshold ({RAG_SCORE_THRESHOLD})")
        return []

    selected = top_results[0]
    snippet = (selected.get("chunk_text") or "")[:200].replace("\n", " ")
    print(
        f"[RAG] Final selected chunk: department={selected.get('department')} "
        f"score={selected.get('score', 0.0):.4f} text='{snippet}'"
    )

    return top_results
    
def build_context(query: str, top_k: int = 3):
    results = search_knowledge_base(query, top_k)

    if not results:
        return None

    context_blocks = []
    for r in results:
        context_blocks.append(r["chunk_text"])

    return "\n\n---\n\n".join(context_blocks)

if __name__ == "__main__":
    query = "If urgent client demand requires hiring beyond approved headcount at senior level, what approvals and documentation are required?"
    results = search_knowledge_base(query)

    print("\nTop Results:\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i} | Department: {result['department']} | Score: {result['score']:.4f}")
        print(result["chunk_text"][:500])
        print("-" * 80)
