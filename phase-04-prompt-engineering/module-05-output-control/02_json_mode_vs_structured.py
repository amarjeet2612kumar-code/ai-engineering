import json
from typing import Literal

import ollama
from pydantic import BaseModel, ValidationError


MODEL = "llama3.2:3b"


# =========================================================
# APPLICATION CONTRACT
#
# This is what our application actually expects.
# =========================================================

class IncidentResult(BaseModel):

    severity: Literal[
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]

    requires_approval: bool


# =========================================================
# INPUT
# =========================================================

incident = """
Spark job CRM360-123 failed.

ExecutorLostFailure occurred.

Executor memory utilization reached 96%.

The failure happened during the aggregation stage.
"""


# =========================================================
# EXPERIMENT 1
#
# Ask for JSON without using JSON mode.
# =========================================================

prompt_1 = f"""
Analyze this incident:

{incident}

Return the result as JSON with:

severity
requires_approval

Do not provide any explanation.
"""


response_1 = ollama.chat(
    model=MODEL,

    messages=[
        {
            "role": "user",
            "content": prompt_1
        }
    ]
)


raw_1 = response_1["message"]["content"]


print("\n========================================")
print("EXPERIMENT 1 — PROMPT ONLY")
print("========================================")

print(raw_1)


# =========================================================
# EXPERIMENT 2
#
# Ask Ollama to generate JSON.
# =========================================================

#if uncomment prompt_2 :- JSON Mode may guarantee that the response is syntactically valid JSON, but pydentic failed.

# prompt_2 = f"""
# Analyze this incident:

# {incident}

# Return ONLY JSON.

# Use these fields:

# severity
# requires_approval
# """

prompt_2 = f"""
Analyze this incident:

{incident}

Return ONLY JSON.

Use these fields:

severity
requires_approval

For severity, choose one of:
LOW, MEDIUM, HIGH, CRITICAL.

requires_approval must be true or false.
"""

response_2 = ollama.chat(
    model=MODEL,

    messages=[
        {
            "role": "user",
            "content": prompt_2
        }
    ],

    format="json"
)


raw_2 = response_2["message"]["content"]


print("\n========================================")
print("EXPERIMENT 2 — JSON MODE")
print("========================================")

print(raw_2)


# =========================================================
# PARSE JSON
# =========================================================

try:

    data = json.loads(raw_2)

    print("\nJSON parsing: SUCCESS")

except json.JSONDecodeError as e:

    print("\nJSON parsing: FAILED")

    print(e)

    raise SystemExit


# =========================================================
# VALIDATE AGAINST APPLICATION CONTRACT
# =========================================================

print("\n========================================")
print("SCHEMA VALIDATION")
print("========================================")


try:

    result = IncidentResult.model_validate(data)

    print("Schema validation: SUCCESS")

    print(result)


except ValidationError as e:

    print("Schema validation: FAILED")

    print(e)