from litellm import completion

response = completion(
    model="ollama/qwen2.5:7b",
    messages=[
        {
            "role": "user",
            "content": "Explain Spark partitioning in one sentence."
        }
    ]
)

print(response.choices[0].message.content)