import json
import hashlib
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents.middleware import ModelRequest, ToolCallRequest
from Database.session import get_connection
from Config.model import get_embeddings
from langgraph.config import get_config
import numpy as np

# Semantic Cache
def get_most_recent_user_query(messages: List[Any]) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content")
    return None

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@wrap_model_call
def wrap_semantic_cache(request: ModelRequest, handler: Callable):
    messages = request.messages
    model = request.model
    model_name = getattr(model, "model_name", "unknown") if hasattr(model, "model_name") else str(model)

    query = get_most_recent_user_query(messages)
    if not query:
        return handler(request)

    embeddings = get_embeddings()
    query_vector = embeddings.embed_query(query)
    query_vector = np.array(query_vector)

    conn = get_connection()
    cur = conn.cursor()

    best_match = None
    best_score = 0.0

    try:
        cur.execute("""
            SELECT embedding, ai_response
            FROM semantic_cache
        """)
        rows = cur.fetchall()

        for row in rows:
            stored_embedding = np.array(row[0])
            ai_response = row[1]

            score = cosine_similarity(query_vector, stored_embedding)

            if score > best_score:
                best_score = score
                best_match = ai_response

        if best_score > 0.80 and best_match:
            config = get_config()
            config.setdefault("metadata", {})
            config["metadata"]["cache_hit"] = True

            cur.close()
            conn.close()
            return AIMessage(content=best_match)

    except Exception as e:
        print(f"Cache lookup error: {e}")
    finally:
        if not conn.closed:
            cur.close()
            conn.close()

    # Call LLM
    response = handler(request)

    # Store only final response
    if isinstance(response, AIMessage) and not response.tool_calls:
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO semantic_cache (embedding, original_query, ai_response, model_name)
                VALUES (%s, %s, %s, %s)
            """, (query_vector.tolist(), query, response.content, model_name))
            conn.commit()
        except Exception as e:
            print(f"Cache store error: {e}")
        finally:
            cur.close()
            conn.close()

    return response

# Tool Cache
CACHEABLE_TOOLS = [
    "lookup_policy",
    "get_ticket",
    "get_user_ticket_history",
    "get_ticket_status"
]

@wrap_tool_call
def wrap_tool_cache(request: ToolCallRequest, handler: Callable):
    tool_name = request.tool.name if hasattr(request.tool, "name") else str(request.tool)
    args = request.tool_call["args"]
    
    if tool_name not in CACHEABLE_TOOLS:
        return handler(request)
    
    args_str = json.dumps(args, sort_keys=True)
    args_hash = hashlib.sha256(args_str.encode()).hexdigest()
    
    # Update metadata for logging (counting tool calls)
    if hasattr(request.runtime, "config") and "metadata" in request.runtime.config:
         request.runtime.config["metadata"]["tool_calls_count"] = request.runtime.config["metadata"].get("tool_calls_count", 0) + 1

    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT output FROM tool_cache
            WHERE tool_name = %s AND args_hash = %s
        """, (tool_name, args_hash))
        
        match = cur.fetchone()
        if match:
            cur.close()
            conn.close()
            return match[0] # output is JSONB/dict
            
    except Exception as e:
        print(f"Tool cache lookup error: {e}")
    finally:
        if not conn.closed:
            cur.close()
            conn.close()

    # Execute tool
    result = handler(request)
    
    # Store result
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO tool_cache (tool_name, args_hash, output)
            VALUES (%s, %s, %s)
            ON CONFLICT (tool_name, args_hash) DO UPDATE SET output = EXCLUDED.output, created_at = CURRENT_TIMESTAMP
        """, (tool_name, args_hash, json.dumps(result)))
        conn.commit()
    except Exception as e:
        print(f"Tool cache store error: {e}")
    finally:
        cur.close()
        conn.close()
        
    return result

