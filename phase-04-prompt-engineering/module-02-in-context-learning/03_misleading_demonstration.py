import ollama


prompt = """
Classify DataOps incidents into:

OUT_OF_MEMORY
KAFKA_LAG
TIMEOUT
PERMISSION_ERROR


Example 1:

Input:
Spark executor exceeded its memory limit.

Output:
OUT_OF_MEMORY


Example 2:

Input:
Kafka consumer is behind the producer.

Output:
KAFKA_LAG


Example 3:

Input:
Spark executor exceeded its memory limit.

Output:
TIMEOUT


Now classify:

Input:
Spark executor was killed because it exceeded
the configured memory limit.

Output:
"""


response = ollama.chat(
    model="llama3.2:3b",

    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


print("========== MISLEADING DEMONSTRATION ==========")

print(response["message"]["content"])