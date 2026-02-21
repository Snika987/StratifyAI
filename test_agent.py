from Agents.agent import agent
from uuid import uuid4

thread_id = "user_123"

response = agent.invoke(
    {"messages": [{"role": "user", "content": "My VM is not working"}]},
    config={
        "configurable": {
            "thread_id": "thread_1",
            "user_id": str(uuid4())
        }
    }
)

print(response)


