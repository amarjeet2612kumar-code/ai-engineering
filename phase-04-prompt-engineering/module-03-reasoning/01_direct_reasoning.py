import ollama


# =========================================================
# DIRECT PROMPT
#
# We ask the model for the answer directly.
# We do NOT explicitly ask for intermediate reasoning.
# =========================================================

problem = """
A Spark job processes 240 GB of data.

The cluster has 6 executors.

Each executor can process 8 GB per hour.

During processing, 25% of the cluster capacity
is lost because of overhead.

How many hours are required to process the job?

Return only the final answer.
"""


response = ollama.chat(
    model="llama3.2:3b",

    messages=[
        {
            "role": "user",
            "content": problem
        }
    ]
)


print("========== DIRECT ANSWER ==========")

print(response["message"]["content"])