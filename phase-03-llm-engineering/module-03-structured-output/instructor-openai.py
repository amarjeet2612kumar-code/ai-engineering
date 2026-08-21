# ---------------------------------------------------------
# Import required libraries
# ---------------------------------------------------------

from typing import Literal

import instructor
from openai import OpenAI

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------
# 1. Define the structure we EXPECT from the LLM
# ---------------------------------------------------------
#
# Pydantic model acts as our response schema.
#
# The LLM must provide:
#
# error_type          -> string
# severity            -> LOW / MEDIUM / HIGH / CRITICAL
# recommended_action  -> string
# requires_approval   -> boolean
#
# ---------------------------------------------------------

class JobFailure(BaseModel):

    # strict=True means Pydantic will strictly check
    # the data types instead of accepting conversions.
    #
    # Example:
    # requires_approval = "yes"
    #
    # Expected -> bool
    # Received -> str
    #
    # Therefore validation will fail.
    model_config = ConfigDict(strict=True)

    error_type: str

    # Literal means only these values are allowed.
    #
    # Valid:
    # LOW
    # MEDIUM
    # HIGH
    # CRITICAL
    #
    # Invalid:
    # NORMAL
    # BANANA
    # 1
    #
    severity: Literal[
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]

    recommended_action: str

    requires_approval: bool


# ---------------------------------------------------------
# 2. Create an OpenAI-compatible client
# ---------------------------------------------------------
#
# Ollama provides an OpenAI-compatible API endpoint.
#
# Ollama normally runs on:
#
# http://localhost:11434
#
# Its OpenAI-compatible endpoint is:
#
# http://localhost:11434/v1
#
# ---------------------------------------------------------

client = OpenAI(
    base_url="http://localhost:11434/v1",

    # Ollama does not require a real OpenAI API key.
    # We just provide a dummy value because the OpenAI
    # Python client expects an API key.
    api_key="ollama"
)


# ---------------------------------------------------------
# 3. Give the OpenAI-compatible client to Instructor
# ---------------------------------------------------------
#
# instructor.from_openai() wraps the client.
#
# Its purpose:
#
# "Allow Instructor to use Pydantic models when
# communicating with the LLM."
#
# ---------------------------------------------------------

client = instructor.from_openai(client)


# ---------------------------------------------------------
# 4. Call the LLM
# ---------------------------------------------------------
#
# response_model=JobFailure is the MOST IMPORTANT part.
#
# It tells Instructor:
#
# "I expect the LLM response to follow the
# JobFailure Pydantic model."
#
# Instructor then handles the structured-output
# and validation workflow.
#
# ---------------------------------------------------------

try:

    failure = client.chat.completions.create(

        # This is your local Ollama model.
        model="llama3.2:3b",

        # Prompt sent to the LLM.
        messages=[
            {
                "role": "user",

                "content": """
                Analyze this DataOps incident:

                Spark job failed because executor
                ran out of memory.

                Determine:

                1. error_type
                2. severity
                3. recommended_action
                4. requires_approval
                """
            }
        ],

        # Tell Instructor which Pydantic model
        # should be used for the response.
        response_model=JobFailure
    )


    # -----------------------------------------------------
    # 5. If we reach here, structured validation succeeded
    # -----------------------------------------------------

    print("\n========== SUCCESS ==========")

    print("LLM returned a valid JobFailure object.")

    # Print the complete Pydantic object.
    print("\nComplete response:")
    print(failure)


    # -----------------------------------------------------
    # 6. Access individual fields
    # -----------------------------------------------------

    print("\nIndividual fields:")

    print("Error Type         :", failure.error_type)

    print("Severity           :", failure.severity)

    print(
        "Recommended Action :",
        failure.recommended_action
    )

    print(
        "Requires Approval  :",
        failure.requires_approval
    )


# ---------------------------------------------------------
# 7. Handle failure gracefully
# ---------------------------------------------------------
#
# If the LLM response cannot be converted into the
# JobFailure structure, Instructor/Pydantic can raise
# an exception.
#
# Instead of allowing the application to crash,
# we catch the error and print a message.
#
# ---------------------------------------------------------

except Exception as e:

    print("\n========== FAILED ==========")

    print("❌ Could not create a valid JobFailure response.")

    print("\nError:")
    print(e)