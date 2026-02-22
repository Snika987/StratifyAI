from Agents.graph import invoke_agent
import uuid

if __name__ == "__main__":
    result = invoke_agent(
        user_query="My VM is not working and I need urgent help.",
        thread_id="thread_2",
        user_id = str(uuid.uuid4()),
    )
    print(result)
