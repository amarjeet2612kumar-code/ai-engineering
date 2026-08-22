import ollama


# =========================================================
# BAD DEMONSTRATIONS
#
# These examples are:
# - vague
# - inconsistent
# - not representative
# - using unclear output labels
# =========================================================

prompt = """
Classify the following DataOps incidents.

Examples:

Example 1:
Input:
Something went wrong with Spark.

Output:
PROBLEM


Example 2:
Input:
Kafka was slow.

Output:
BAD


Example 3:
Input:
The pipeline did not work.

Output:
ERROR


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


print("========== BAD DEMONSTRATIONS ==========")

print(response["message"]["content"])