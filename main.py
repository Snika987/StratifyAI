from Agents.graph import invoke_agent
import uuid

if __name__ == "__main__":
    thread_id = "thread_test"
    user_id = "11111111-1111-1111-1111-111111111111"

    # print("---- TURN 1 ----")
    # result1 = invoke_agent(
    #     user_query="Install VS 2024 and my VPN is not working",
    #     thread_id=thread_id,
    #     user_id=user_id,
    # )
    # print(result1)

    print("\n---- TURN 2 ----")
    result2 = invoke_agent(
        user_query="yes",
        thread_id=thread_id,
        user_id=user_id,
    )
    print(result2)

print("THREAD:", thread_id)
print("USER:", user_id)