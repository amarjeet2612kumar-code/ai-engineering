import ollama

prompt = """
Classify the sentiment of the text as Positive or Negative.

Example:
Text: "I really enjoyed this movie."
Sentiment: Positive

Now classify:

Text: "The movie was boring and disappointing."
Sentiment:
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