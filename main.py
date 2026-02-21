from agents.support_agent import agent

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "My VM is not working"}
    ]
})

print(response)
