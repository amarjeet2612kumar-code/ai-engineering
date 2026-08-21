import ollama

prompts = {
    "poor": """
Explain Spark partitioning.
""",

    "clear": """
You are a senior Apache Spark engineer.

Explain Spark partitioning to a data engineer
who already understands basic PySpark.

Cover:
1. What partitioning means
2. Why partitioning matters
3. One practical example

Keep the answer under 150 words.
Use simple language.
"""
}

for name, prompt in prompts.items():
    print(f"\n{'=' * 20} {name.upper()} {'=' * 20}")

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