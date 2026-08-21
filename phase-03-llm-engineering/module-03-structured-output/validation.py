import json
import ollama

from typing import Literal
from pydantic import BaseModel, ConfigDict, ValidationError


# --------------------------------------------------
# 1. Define the expected LLM response structure
# --------------------------------------------------

class JobFailure(BaseModel):

    # Purpose:
    # Enforce strict type validation.
    model_config = ConfigDict(strict=True)

    error_type: str

    # Purpose:
    # Allow only these four severity values.
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    recommended_action: str

    requires_approval: bool


# --------------------------------------------------
# 2. Call the LLM
# --------------------------------------------------

response = ollama.chat(
    model="llama3.2:3b",

    messages=[
        {
            "role": "user",
            "content": """
            Analyze this DataOps incident:

            Spark job failed because executor
            ran out of memory.

            Return JSON containing exactly:

            error_type
            severity
            recommended_action
            requires_approval

            severity must be one of:
            LOW, MEDIUM, HIGH, CRITICAL

            requires_approval must be true or false.
            """
        }
    ],

    # Purpose:
    # Tell Ollama to return JSON instead of normal text.
    format="json"
)


# --------------------------------------------------
# 3. Extract the LLM response
# --------------------------------------------------

content = response["message"]["content"]

print("========== LLM RESPONSE ==========")
print(content)


# --------------------------------------------------
# 4. Convert JSON string → Python dictionary
# --------------------------------------------------

try:

    # Purpose:
    # Convert JSON text returned by the LLM into a Python dictionary.
    data = json.loads(content)

    print("\n========== PARSED DATA ==========")
    print(data)

except json.JSONDecodeError as e:

    print("\n❌ LLM returned invalid JSON")
    print(e)

    exit()


# --------------------------------------------------
# 5. Validate using Pydantic
# --------------------------------------------------

try:

    # Purpose:
    # Validate the dictionary against JobFailure
    # and create a JobFailure Python object.
    failure = JobFailure.model_validate(data)

    print("\n========== VALIDATION SUCCESS ==========")

    print("✅ LLM output is valid")

    print("\nValidated object:")
    print(failure)

    print("\nIndividual fields:")
    print("Error Type         :", failure.error_type)
    print("Severity           :", failure.severity)
    print("Recommended Action :", failure.recommended_action)
    print("Requires Approval  :", failure.requires_approval)


except ValidationError as e:

    print("\n========== VALIDATION FAILED ==========")

    print("❌ LLM output does not match JobFailure schema")

    print("\nValidation details:")
    print(e)