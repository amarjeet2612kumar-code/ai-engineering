import ollama


# =========================================================
# GOOD DEMONSTRATIONS
#
# Each example:
# 1. Clearly describes the incident
# 2. Uses the same classification scheme
# 3. Has a meaningful output
# 4. Represents the task we want to perform
# =========================================================

prompt = """
Classify each DataOps incident into exactly one category:

OUT_OF_MEMORY
KAFKA_LAG
TIMEOUT
PERMISSION_ERROR


Example 1:

Input:
Spark executor was terminated because it exceeded
the configured memory limit.

Output:
OUT_OF_MEMORY


Example 2:

Input:
Kafka consumer lag increased significantly because
the consumer could not process messages fast enough.

Output:
KAFKA_LAG


Example 3:

Input:
Airflow task exceeded the configured execution timeout.

Output:
TIMEOUT


Example 4:

Input:
Spark application failed because the service account
did not have permission to access the required data.

Output:
PERMISSION_ERROR


Now classify this incident:

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


print("========== GOOD DEMONSTRATIONS ==========")

print(response["message"]["content"])