import json
import ollama
from pydantic import BaseModel, ValidationError


class SparkIncident(BaseModel):
    error_type: str
    root_cause: str
    priority: str


# Call LLM
response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": """
            Analyze this incident:

            Spark job failed because executor
            ran out of memory.

            Return JSON containing:
            error_type,
            root_cause,
            priority.
            """
        }
    ],
    format="json"
)


# Get LLM response
content = response["message"]["content"]

print("========== LLM RESPONSE ==========")
print(content)


# Convert JSON string → Python dictionary
try:
    data = json.loads(content)

except json.JSONDecodeError as e:
    print("\n❌ Invalid JSON returned by LLM")
    print(f"Error: {e}")
    exit()


print("\n========== PARSED JSON ==========")
print(data)


# Validate using Pydantic
try:
    incident = SparkIncident.model_validate(data)

    print("\n========== VALIDATION SUCCESS ==========")
    print("✅ LLM output is valid")

    print("\nIncident:")
    print(incident)

    print("\nFields:")
    print("Error Type :", incident.error_type)
    print("Root Cause :", incident.root_cause)
    print("Priority   :", incident.priority)


except ValidationError as e:
    print("\n========== VALIDATION FAILED ==========")
    print("❌ LLM output does not match SparkIncident schema")

    print("\nValidation details:")
    print(e)