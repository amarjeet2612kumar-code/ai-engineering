import ollama

template = """
Explain {topic} to a {audience}.

Requirements:
- Use simple language.
- Give one practical example.
- Keep the answer under 150 words.
"""

topic = "Spark partitioning"
audience = "data engineer"

prompt = template.format(
    topic=topic,
    audience=audience
)

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response["message"]["content"])