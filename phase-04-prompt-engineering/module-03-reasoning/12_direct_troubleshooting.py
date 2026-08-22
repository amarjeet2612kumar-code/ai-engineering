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
# DIRECT APPROACH
#
# We immediately ask the LLM to diagnose the incident.
# =========================================================

response = ollama.chat(
    model="llama3.2:3b",

    messages=[
        {
            "role": "user",
            "content": incident
        }
    ]
)


print("========== DIRECT APPROACH ==========")

print(
    response["message"]["content"]
)