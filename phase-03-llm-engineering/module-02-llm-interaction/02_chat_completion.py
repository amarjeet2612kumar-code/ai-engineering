import ollama

messages = [
    {
        "role": "system",
        "content": "You are an Apache Spark expert."
    }
]

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    response = ollama.chat(
        model="llama3.2:3b",
        messages=messages
    )

    assistant_message = response["message"]["content"]

    print(f"Assistant: {assistant_message}")

    messages.append({
        "role": "assistant",
        "content": assistant_message
    })