import ollama


# =========================================================
# SAME INFORMATION FOR EVERY EXPERIMENT
#
# We deliberately keep the information identical.
# Only the ORDER will change.
# =========================================================

context = """
Order 101: COMPLETE
Order 102: PENDING
Order 103: COMPLETE
"""


question = """
Which orders are currently pending?
"""


instruction = """
Answer the user's question using only the provided
order information.

Do not invent any additional orders.
"""


# =========================================================
# EXPERIMENT 1
#
# Context → Question
# =========================================================

prompt_1 = f"""
{context}

{question}
"""


# =========================================================
# EXPERIMENT 2
#
# Question → Context
# =========================================================

prompt_2 = f"""
{question}

{context}
"""


# =========================================================
# EXPERIMENT 3
#
# Instruction → Context → Question
# =========================================================

prompt_3 = f"""
{instruction}

Relevant information:

{context}

User question:

{question}
"""


# =========================================================
# Run experiments
# =========================================================

prompts = [
    ("Context → Question", prompt_1),
    ("Question → Context", prompt_2),
    ("Instruction → Context → Question", prompt_3)
]


for name, prompt in prompts:

    response = ollama.chat(
        model="llama3.2:3b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\n========================================")
    print(name)
    print("========================================")

    print("\nPrompt:")
    print(prompt)

    print("\nLLM Response:")
    print(
        response["message"]["content"]
    )