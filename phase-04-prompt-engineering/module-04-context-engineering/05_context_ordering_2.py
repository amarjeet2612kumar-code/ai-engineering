import ollama


# =========================================================
# SAME INFORMATION FOR EVERY EXPERIMENT
#
# We deliberately keep the information identical.
# Only the ORDER will change.
# =========================================================

context = """
Customer and order information:

Customer 101:
Name: Rahul
Location: Bangalore
Preferred category: Football

Customer 102:
Name: Priya
Location: Mumbai
Preferred category: Books

Customer 103:
Name: Arjun
Location: Delhi
Preferred category: Cricket


Order 101:
Date: 2026-01-01
Customer: 101
Status: COMPLETE

Order 102:
Date: 2026-01-02
Customer: 102
Status: PENDING

Order 103:
Date: 2026-01-03
Customer: 103
Status: COMPLETE


System information:

Spark cluster:
3 workers

Kafka:
12 partitions

Airflow:
15 DAGs
"""

question = """
Which customer's order is currently pending?
Return the customer ID and order ID.
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