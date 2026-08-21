import ollama

prompt = """
Classify each text as Positive or Negative.

Example 1:
Text: "The product is excellent."
Sentiment: Positive

Example 2:
Text: "The service was terrible."
Sentiment: Negative

Example 3:
Text: "I am very happy with the purchase."
Sentiment: Positive

Now classify:

Text: "The delivery experience was disappointing."
Sentiment:

Return only Positive or Negative.
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