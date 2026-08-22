import ollama


# =========================================================
# NORMAL ZERO-SHOT
#
# No examples.
# No reasoning instruction.
# =========================================================

problem = """
Three jobs A, B, and C must run one after another.

Rules:

1. A must run before B.
2. C must run after A.
3. B must run before C.

What is the execution order?
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


print("========== ZERO-SHOT ==========")

print(response["message"]["content"])