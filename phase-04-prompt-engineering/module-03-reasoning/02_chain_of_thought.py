import ollama


# =========================================================
# CHAIN-OF-THOUGHT STYLE PROMPT
#
# We explicitly ask the model to work through the
# calculation step by step before giving the answer.
# =========================================================

problem = """
A Spark job processes 240 GB of data.

The cluster has 6 executors.

Each executor can process 8 GB per hour.

During processing, 25% of the cluster capacity
is lost because of overhead.

How many hours are required to process the job?

Work through the calculation step by step,
then provide the final answer.
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


print("========== CHAIN-OF-THOUGHT STYLE ANSWER ==========")

print(response["message"]["content"])