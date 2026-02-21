from Database.checkpointer import get_checkpointer

with get_checkpointer() as checkpointer:
    checkpointer.setup()
