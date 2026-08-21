import ollama

models = [
    "llama3.2:3b",
    "qwen2.5:7b",
]

prompt = """
Explain Spark partitioning to a data engineer
in exactly 3 sentences.
"""

for model in models:

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print(f"\n===== {model} =====")
    print(response["message"]["content"])