import time
import json
from datetime import datetime
from typing import Any, Dict, Optional
from Database.session import get_connection
from langchain.agents.middleware import before_agent, before_model, after_model, after_agent
from langgraph.config import get_config

@before_agent
def before_agent_hook(state: Any, runtime: Any):
    config = get_config()
    print(f"DEBUG: before_agent_hook called. State type: {type(state)}")
    # Initialize logging context in metadata
    if "metadata" not in config:
        config["metadata"] = {}
    
    config["metadata"]["agent_start_time"] = time.time()
    config["metadata"]["tool_calls_count"] = 0
    config["metadata"]["model_calls_count"] = 0
    config["metadata"]["total_tokens"] = {"input": 0, "output": 0, "total": 0}
    config["metadata"]["retries"] = 0
    config["metadata"]["cache_hit"] = False
    
    # Capture raw input
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        config["metadata"]["raw_input"] = getattr(last_msg, "content", str(last_msg))

@before_model
def before_model_hook(state: Any, runtime: Any):
    config = get_config()
    if "metadata" not in config:
        config["metadata"] = {}
    config["metadata"]["model_start_time"] = time.time()

@after_model
def after_model_hook(state: Any, runtime: Any):
    config = get_config()

    # Defensive initialization
    metadata = config.setdefault("metadata", {})
    metadata.setdefault("total_tokens", {"input": 0, "output": 0, "total": 0})
    metadata.setdefault("tool_calls_count", 0)
    metadata.setdefault("model_calls_count", 0)

    # Latency tracking
    latency = time.time() - metadata.get("model_start_time", time.time())
    metadata["model_latency"] = metadata.get("model_latency", 0) + latency
    metadata["model_calls_count"] += 1

    # Get latest model message
    messages = state.get("messages", [])
    if not messages:
        return

    output = messages[-1]

    # Token tracking
    if hasattr(output, "response_metadata") and output.response_metadata:
        usage = output.response_metadata.get("token_usage", {})
        current = metadata["total_tokens"]
        current["input"] += usage.get("prompt_tokens", 0)
        current["output"] += usage.get("completion_tokens", 0)
        current["total"] += usage.get("total_tokens", 0)

    # Tool call tracking
    if hasattr(output, "tool_calls") and output.tool_calls:
        metadata["tool_calls_count"] += len(output.tool_calls)

@after_agent
def after_agent_hook(state: Any, runtime: Any):
    config = get_config()
    end_time = time.time()
    metadata = config.get("metadata", {})
    start_time = metadata.get("agent_start_time", end_time)
    total_latency = end_time - start_time
    
    thread_id = config.get("configurable", {}).get("thread_id")
    user_id = config.get("configurable", {}).get("user_id", "anonymous")
    
    log_data = {
        "thread_id": thread_id,
        "user_id": user_id,
        "raw_input": metadata.get("raw_input"),
        "model_name": metadata.get("model_name", "mistral-large-latest"),
        "token_usage": json.dumps(metadata.get("total_tokens")),
        "tool_calls": metadata.get("tool_calls_count"),
        "latency": total_latency,
        "cache_hit": metadata.get("cache_hit", False),
        "retry_count": metadata.get("retries", 0),
        "status": "success" # Assuming success if we reached here
    }
    
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO logs (thread_id, user_id, raw_input, model_name, token_usage, tool_calls, latency, cache_hit, retry_count, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            log_data["thread_id"],
            log_data["user_id"],
            log_data["raw_input"],
            log_data["model_name"],
            log_data["token_usage"],
            log_data["tool_calls"],
            log_data["latency"],
            log_data["cache_hit"],
            log_data["retry_count"],
            log_data["status"]
        ))
        conn.commit()
    except Exception as e:
        print(f"Logging error: {e}")
    finally:
        cur.close()
        conn.close()

