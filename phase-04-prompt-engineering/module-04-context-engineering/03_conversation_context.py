import ollama


# =========================================================
# Conversation history
# =========================================================

messages = []


# =========================================================
# First message
# =========================================================

messages.append(
    {
        "role": "user",
        "content": "We are building a DataOps platform."
    }
)

response = ollama.chat(
    model="llama3.2:3b",
    messages=messages
)

messages.append(
    {
        "role": "assistant",
        "content": response["message"]["content"]
    }
)


print("========== TURN 1 ==========")

print(
    response["message"]["content"]
)


# =========================================================
# Second message
# =========================================================

messages.append(
    {
        "role": "user",
        "content": "The platform monitors Spark jobs."
    }
)

response = ollama.chat(
    model="llama3.2:3b",
    messages=messages
)

messages.append(
    {
        "role": "assistant",
        "content": response["message"]["content"]
    }
)


print("\n========== TURN 2 ==========")

print(
    response["message"]["content"]
)


# =========================================================
# Third message
# =========================================================

messages.append(
    {
        "role": "user",
        "content": "What technology should analyze the Spark failures?"
    }
)

response = ollama.chat(
    model="llama3.2:3b",
    messages=messages
)


print("\n========== TURN 3 ==========")

print(
    response["message"]["content"]
)


# =========================================================
# Show conversation size
# =========================================================

print("\n========== CONTEXT INFORMATION ==========")

print(
    "Number of messages:",
    len(messages)
)

print(
    "Input tokens in final request:",
    response.get("prompt_eval_count")
)

print(
    "Output tokens in final request:",
    response.get("eval_count")
)