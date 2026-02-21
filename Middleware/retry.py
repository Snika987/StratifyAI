import time
import random
from typing import Callable, Any
from langchain.agents.middleware import wrap_model_call, ModelRequest

from langgraph.config import get_config

@wrap_model_call
def wrap_retry(request: ModelRequest, handler: Callable):
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries + 1):
        try:
            return handler(request)
        except Exception as e:
            # Simple detection of transient errors
            error_msg = str(e).lower()
            is_transient = any(term in error_msg for term in [
                "timeout", "rate limit", "429", "500", "502", "503", "504", 
                "connection", "server error"
            ])
            
            # Check for validation/logic errors which shouldn't be retried
            is_validation = any(term in error_msg for term in [
                "validation", "invalid", "bad request", "400", "permission"
            ])
            
            if is_transient and not is_validation and attempt < max_retries:
                # Update retry count in metadata
                config = get_config()
                if "metadata" in config:
                    config["metadata"]["retries"] = config["metadata"].get("retries", 0) + 1
                
                # Exponential backoff: base * 2^attempt + jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Transient error detected: {e}. Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                # Not a transient error or max retries reached
                raise e

