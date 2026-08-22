import ollama


# =========================================================
# ZERO-SHOT CHAIN-OF-THOUGHT
#
# There are still NO examples.
#
# The only difference is that we explicitly ask the
# model to reason through the constraints.
# =========================================================

problem = """
Three jobs A, B, and C must run one after another.

Rules:

1. A must run before B.
2. C must run after A.
3. B must run before C.

Determine the execution order.

Reason through the constraints step by step,
then provide the final execution order.
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


print("========== ZERO-SHOT CoT ==========")

print(response["message"]["content"])