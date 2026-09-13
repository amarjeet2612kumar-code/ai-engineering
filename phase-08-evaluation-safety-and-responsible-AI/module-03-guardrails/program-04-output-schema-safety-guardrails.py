import json  # Provides JSON parsing for structured LLM output.
from dotenv import load_dotenv  # Loads environment variables from the .env file.
from openai import OpenAI  # Provides the OpenAI API client.
from pydantic import BaseModel, Field, ValidationError  # Provides structured output validation.


load_dotenv()  # Loads OPENAI_API_KEY from the .env file.

client = OpenAI()  # Creates the OpenAI API client.

MODEL_NAME = "gpt-5-mini"  # Defines the LLM used for generating structured output.


class SparkDiagnosis(BaseModel):
    # Defines the required structure for a Spark diagnosis.
    root_cause: str = Field(min_length=1)

    # Requires the model to provide a severity classification.
    severity: str = Field(pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")

    # Requires recommended troubleshooting actions.
    recommendations: list[str] = Field(min_length=1)

    # Indicates whether human review is recommended.
    requires_human_review: bool


def validate_schema(raw_output):
    # Attempts to parse the LLM output as JSON.
    try:
        parsed_output = json.loads(raw_output)

    # Handles invalid JSON responses.
    except json.JSONDecodeError as error:
        return {
            "valid": False,
            "stage": "JSON_VALIDATION",
            "reason": f"Invalid JSON: {error}",
        }

    # Attempts to validate the JSON against the Pydantic schema.
    try:
        diagnosis = SparkDiagnosis.model_validate(parsed_output)

    # Handles missing fields, wrong types, or invalid values.
    except ValidationError as error:
        return {
            "valid": False,
            "stage": "SCHEMA_VALIDATION",
            "reason": error.errors(),
        }

    # Returns the validated Pydantic object.
    return {
        "valid": True,
        "stage": "SCHEMA_VALIDATION",
        "data": diagnosis,
    }


def validate_safety(diagnosis):
    # Blocks critical actions from being automatically recommended.
    dangerous_actions = [
        "delete production data",
        "drop production table",
        "terminate production cluster",
        "delete production database",
    ]

    # Combines all recommendations into one lowercase string.
    recommendations_text = " ".join(diagnosis.recommendations).lower()

    # Checks whether a dangerous production action appears in the recommendations.
    for action in dangerous_actions:
        if action in recommendations_text:
            return {
                "valid": False,
                "stage": "SAFETY_VALIDATION",
                "reason": f"Dangerous action detected: {action}",
            }

    # Requires human review for high or critical severity.
    if diagnosis.severity in {"HIGH", "CRITICAL"}:
        if diagnosis.requires_human_review is not True:
            return {
                "valid": False,
                "stage": "SAFETY_VALIDATION",
                "reason": (
                    "High-risk diagnosis must require human review"
                ),
            }

    # Returns success when the safety policy passes.
    return {
        "valid": True,
        "stage": "SAFETY_VALIDATION",
        "reason": "Output passed safety validation",
    }


def call_llm(user_input):
    # Defines the expected JSON structure in the model instructions.
    instructions = """
You are a secure DataOps Copilot.

Analyze the Spark problem provided by the user.

Return ONLY valid JSON.

The JSON must contain exactly these fields:
{
  "root_cause": "string",
  "severity": "LOW | MEDIUM | HIGH | CRITICAL",
  "recommendations": ["string"],
  "requires_human_review": true_or_false
}

Never recommend deleting production data or terminating production
infrastructure automatically.

HIGH and CRITICAL severity must set requires_human_review to true.
"""

    # Sends the user request to the LLM.
    response = client.responses.create(
        model=MODEL_NAME,
        instructions=instructions,
        input=user_input,
    )

    # Returns the raw model output.
    return response.output_text


def run_test(test_name, user_input):
    # Prints the name of the test.
    print("\n" + "=" * 70)
    print(test_name)
    print("=" * 70)

    # Prints the user request.
    print("\nUSER INPUT")
    print("-" * 70)
    print(user_input)

    # Calls the LLM.
    raw_output = call_llm(user_input)

    # Prints the raw model output.
    print("\nRAW LLM OUTPUT")
    print("-" * 70)
    print(raw_output)

    # Validates JSON and schema.
    schema_result = validate_schema(raw_output)

    # Prints the schema validation result.
    print("\nSCHEMA VALIDATION")
    print("-" * 70)
    print(schema_result)

    # Stops processing if the JSON/schema is invalid.
    if not schema_result["valid"]:
        print("\nFINAL RESULT")
        print("-" * 70)
        print({
            "status": "BLOCKED",
            "stage": schema_result["stage"],
            "reason": schema_result["reason"],
        })
        return

    # Retrieves the validated Pydantic object.
    diagnosis = schema_result["data"]

    # Runs application-level safety validation.
    safety_result = validate_safety(diagnosis)

    # Prints the safety validation result.
    print("\nSAFETY VALIDATION")
    print("-" * 70)
    print(safety_result)

    # Blocks the output if the safety policy fails.
    if not safety_result["valid"]:
        print("\nFINAL RESULT")
        print("-" * 70)
        print({
            "status": "BLOCKED",
            "stage": safety_result["stage"],
            "reason": safety_result["reason"],
        })
        return

    # Prints the validated structured object.
    print("\nVALIDATED OUTPUT")
    print("-" * 70)
    print(diagnosis.model_dump())

    # Confirms that the output can safely continue to the application.
    print("\nFINAL RESULT")
    print("-" * 70)
    print({
        "status": "ALLOW",
        "stage": "OUTPUT_GUARDRAILS",
        "reason": "Output passed schema and safety validation",
    })


def main():
    # Tests a normal Spark troubleshooting request.
    run_test(
        "TEST 1 - NORMAL SPARK DIAGNOSIS",
        "A Spark job is failing because one executor runs out of memory.",
    )

    # Tests a low-severity Spark problem.
    run_test(
        "TEST 2 - LOW SEVERITY",
        "A Spark job has a minor configuration issue with logging.",
    )

    # Tests a high-severity production problem.
    run_test(
        "TEST 3 - HIGH SEVERITY",
        "A production Spark job is repeatedly failing and affecting downstream processing.",
    )

    # Tests a critical production scenario.
    run_test(
        "TEST 4 - CRITICAL PRODUCTION ISSUE",
        "A critical production Spark failure is causing major business impact.",
    )


if __name__ == "__main__":
    # Starts the test suite when the program is executed directly.
    main()