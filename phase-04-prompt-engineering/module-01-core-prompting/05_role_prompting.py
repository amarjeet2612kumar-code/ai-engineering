import ollama

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {
            "role": "system",
            "content": """
You are an experienced Apache Spark engineer.
Explain technical concepts clearly for a data engineer.
"""
        },
        {
            "role": "user",
            "content": """
Explain Spark partitioning.
"""
        }
    ]
)

print(response["message"]["content"])