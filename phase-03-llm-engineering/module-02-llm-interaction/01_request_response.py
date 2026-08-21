import ollama

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {
            "role": "user",
            "content": "Explain Spark partitioning in one sentence."
        }
    ]
)

print("Full Response:")
print(response)

print("\nGenerated Text:")
print(response["message"]["content"])