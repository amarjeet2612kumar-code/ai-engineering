import ollama


# =========================================================
# SPECIFIC INCIDENT
# =========================================================

incident = """
A Spark job repeatedly fails with:

ExecutorLostFailure

The failure happens during a large data transformation.

What is the likely cause and what should we investigate?
"""


# =========================================================
# STEP 1 — STEP BACK
#
# First identify the general principles/categories
# relevant to this type of failure.
# =========================================================

general_prompt = """
Before diagnosing a specific Spark incident, step back.

What are the major general causes of
ExecutorLostFailure in Apache Spark?

Provide a concise list of the relevant
failure categories.
"""


response = ollama.chat(
    model="llama3.2:3b",

    messages=[
        {
            "role": "user",
            "content": general_prompt
        }
    ]
)


general_principles = response["message"]["content"].strip()


print("========== GENERAL PRINCIPLES ==========")

print(general_principles)


# =========================================================
# STEP 2 — APPLY THE PRINCIPLES
#
# Now give the general knowledge back to the model
# together with the specific incident.
# =========================================================

specific_prompt = f"""
We first identified these general causes of
ExecutorLostFailure:

{general_principles}


Now analyze this specific incident:

{incident}


Using the general causes above, determine:

1. The most likely cause.
2. What evidence should be checked.
3. What investigation should be performed.

Do not assume evidence that has not been provided.
"""


response = ollama.chat(
    model="llama3.2:3b",

    messages=[
        {
            "role": "user",
            "content": specific_prompt
        }
    ]
)


print("\n========== SPECIFIC ANALYSIS ==========")

print(
    response["message"]["content"]
)