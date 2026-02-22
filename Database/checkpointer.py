from pathlib import Path
from dotenv import dotenv_values
from langgraph.checkpoint.memory import InMemorySaver

BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
config = dotenv_values(env_path)


def get_checkpointer():
    return InMemorySaver()