from typing import Literal
import json

import ollama
from pydantic import BaseModel, ValidationError


# =========================================================
# OUTPUT CONTRACT
#
# This defines exactly what our application expects
# from the LLM.
# =========================================================

class JobFailure(BaseModel):

    error_type: str

    severity: Literal[
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]

    recommended_action: str

    requires_approval: bool


# =========================================================
# INPUT TO THE LLM
# =========================================================

job_information = """
Spark Job: CRM360-123

The Spark job failed during the aggregation stage.

Error:
ExecutorLostFailure

Executor 3 reached 96% memory utilization.

Configured executor memory:
16 GB

The current dataset is 2.5 times larger than the
previous successful run.
"""


# =========================================================
# ASK THE LLM FOR STRUCTURED DATA
# =========================================================

prompt = f"""
Analyze the following Spark job failure.

{job_information}

Return ONLY JSON.

The JSON must contain exactly these fields:

- error_type
- severity
- recommended_action
- requires_approval

Allowed severity values:

LOW
MEDIUM
HIGH
CRITICAL

requires_approval must be a boolean.

Do not add explanations outside the JSON.
"""


response = ollama.chat(
    model="llama3.2:3b",

    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],

    # Ask Ollama for JSON-formatted output.
    format="json"
)


# =========================================================
# RAW LLM OUTPUT
# =========================================================

raw_output = response["message"]["content"]


print("\n========== RAW LLM OUTPUT ==========")

print(raw_output)


# =========================================================
# CONVERT JSON TEXT → PYTHON OBJECT
# =========================================================

try:

    data = json.loads(raw_output)

except json.JSONDecodeError as e:

    print("\nInvalid JSON")
    print(e)

    raise SystemExit


# =========================================================
# VALIDATE AGAINST OUR OUTPUT CONTRACT
#
# We already learned Pydantic validation in Phase 3.
# Here we use it as the application contract.
# =========================================================

try:

    failure = JobFailure.model_validate(data)

    print("\n========== VALIDATED OUTPUT ==========")

    print(failure)


except ValidationError as e:

    print("\n========== CONTRACT VIOLATION ==========")

    print(e)

    raise SystemExit


# =========================================================
# APPLICATION LOGIC
#
# Once validated, normal Python code can safely
# consume the structured result.
# =========================================================

print("\n========== APPLICATION LOGIC ==========")

if failure.requires_approval:

    print(
        "Approval required before remediation."
    )

else:

    print(
        "Automatic remediation may be considered."
    )