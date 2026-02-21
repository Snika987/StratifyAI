import os
from dotenv import dotenv_values
from pathlib import Path
from langgraph.checkpoint.postgres import PostgresSaver

BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
config = dotenv_values(env_path)

def get_checkpointer():
    conn_string = (
        f"postgresql://{config.get('DB_USER')}:"
        f"{config.get('DB_PASSWORD')}@"
        f"{config.get('DB_HOST')}:"
        f"{config.get('DB_PORT')}/"
        f"{config.get('DB_NAME')}"
    )
    return PostgresSaver.from_conn_string(conn_string)
