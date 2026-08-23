from typing import Literal

import ollama


# =========================================================
# MODEL
# =========================================================

MODEL = "llama3.2:3b"


# =========================================================
# CONSTRAINT SCHEMA
#
# The model must generate a JSON object containing:
#
# severity:
#   LOW
#   MEDIUM
#   HIGH
#   CRITICAL
#
# This is stronger than simply asking:
# "Return JSON."
# =========================================================

severity_schema = {
    "type": "object",

    "properties": {

        "severity": {
            "type": "string",

            "enum": [
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL"
            ]
        }

    },

    "required": [
        "severity"
    ]
}


# =========================================================
# INCIDENT
# =========================================================

incident = """
Spark job CRM360-123 failed.

ExecutorLostFailure occurred.

Executor memory utilization reached 96%.

The failure happened during the aggregation stage.
"""


# =========================================================
# PROMPT
#
# Notice that the prompt itself is relatively simple.
#
# The schema is doing the important output restriction.
# =========================================================

prompt = f"""
Analyze the following incident.

{incident}

Determine the severity.
"""


# =========================================================
# LLM CALL
#
# format receives a JSON schema.
#
# Ollama can use this schema to constrain the generated
# structured output.
# =========================================================

response = ollama.chat(

    model=MODEL,

    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],

    format=severity_schema
)


# =========================================================
# RESULT
# =========================================================

print("\n========================================")
print("CONSTRAINED OUTPUT")
print("========================================")

print(
    response["message"]["content"]
)