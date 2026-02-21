import json
import numpy as np
from Database.session import get_connection
from Config.model import get_embeddings


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_knowledge_base(query: str, top_k: int = 3):
    embeddings = get_embeddings()
    query_vector = embeddings.embed_query(query)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT department, chunk_text, embedding
        FROM rag_documents
    """)

    rows = cur.fetchall()

    scored_results = []

    for dept, chunk_text, embedding_json in rows:
        stored_embedding = embedding_json
        score = cosine_similarity(query_vector, stored_embedding)

        scored_results.append({
            "department": dept,
            "chunk_text": chunk_text,
            "score": score
        })

    cur.close()
    conn.close()

    # Sort by similarity descending
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    top_results = scored_results[:top_k]

    if not top_results:
        return []

    if top_results[0]["score"] < 0.82:
        return []

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