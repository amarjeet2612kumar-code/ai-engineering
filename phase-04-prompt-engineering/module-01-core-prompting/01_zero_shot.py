import ollama

prompt = """
Classify the following text as Positive or Negative:

"The product quality is excellent."

Return only one word.
"""

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