import ollama

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {
            "role": "system",
            "content": """
You are a SQL expert.
Return only SQL.
Do not provide explanations.
"""
        },
        {
            "role": "user",
            "content": """
Show all customers whose age is greater than 30.
The table is called customers.
"""
        }
    ]
)

print(response["message"]["content"])