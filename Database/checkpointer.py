from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Optional
import atexit

from dotenv import dotenv_values
from langgraph.checkpoint.postgres import PostgresSaver


BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
config = dotenv_values(env_path)

DATABASE_URL = config["DATABASE_URL"]

_LOCK = Lock()
_CHECKPOINTER: Optional[PostgresSaver] = None
_CHECKPOINTER_CTX: Optional[Any] = None
_SETUP_DONE = False


def get_checkpointer() -> PostgresSaver:
    """Return a process-wide PostgresSaver singleton configured for LangGraph persistence."""
    global _CHECKPOINTER, _CHECKPOINTER_CTX, _SETUP_DONE
    with _LOCK:
        if _CHECKPOINTER is None:
            # from_conn_string() returns a context manager in this LangGraph version.
            _CHECKPOINTER_CTX = PostgresSaver.from_conn_string(DATABASE_URL)
            _CHECKPOINTER = _CHECKPOINTER_CTX.__enter__()
        if not _SETUP_DONE:
            _CHECKPOINTER.setup()
            _SETUP_DONE = True
        return _CHECKPOINTER


def _close_checkpointer() -> None:
    global _CHECKPOINTER_CTX
    with _LOCK:
        if _CHECKPOINTER_CTX is not None:
            _CHECKPOINTER_CTX.__exit__(None, None, None)
            _CHECKPOINTER_CTX = None


atexit.register(_close_checkpointer)
