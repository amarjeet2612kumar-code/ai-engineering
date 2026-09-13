"""
Program 7: Garak LLM Vulnerability Scanner

Purpose:
Use the Garak security framework to test an OpenAI LLM
for prompt-injection-related vulnerabilities.

Garak provides:
    - Security probes
    - Adversarial prompts
    - LLM execution
    - Detectors
    - Security reports

This program only acts as a Python runner around Garak.
"""

# Import os to access environment variables such as OPENAI_API_KEY.
import os

# Import subprocess to execute Garak as a separate process.
import subprocess

# Import load_dotenv to load variables from the project's .env file.
from dotenv import load_dotenv


# Load environment variables from the .env file.
load_dotenv()


# Define the OpenAI model that Garak will test.
MODEL_NAME = "gpt-5-mini"

# Define the Garak probe family we want to test.
PROBE_SPEC = "probes.promptinject"

# Generate one model response for each attack attempt.
GENERATIONS = "1"

# Run multiple Garak probe attempts concurrently.
PARALLEL_ATTEMPTS = "8"


# Check whether the OpenAI API key is available.
if not os.getenv("OPENAI_API_KEY"):

    # Stop execution if the API key is not available.
    raise RuntimeError(
        "OPENAI_API_KEY is not available. "
        "Make sure it exists in your .env file."
    )


# Build the Garak command.
garak_command = [
    "python",
    "-m",
    "garak",
    "--target_type",
    "openai",
    "--target_name",
    MODEL_NAME,
    "--spec",
    PROBE_SPEC,
    "--generations",
    GENERATIONS,
    "--parallel_attempts",
    PARALLEL_ATTEMPTS,
]


# Print the program heading.
print("=" * 70)
print("PROGRAM 7 - GARAK LLM VULNERABILITY SCAN")
print("=" * 70)

# Display the target model.
print(f"MODEL              : {MODEL_NAME}")

# Display the Garak probe being used.
print(f"PROBE SPEC         : {PROBE_SPEC}")

# Display the number of generations per attack.
print(f"GENERATIONS        : {GENERATIONS}")

# Display the parallel execution setting.
print(f"PARALLEL ATTEMPTS  : {PARALLEL_ATTEMPTS}")

# Clearly identify the security framework.
print("FRAMEWORK          : Garak")

# Explain what this program is testing.
print("TEST LEVEL         : LLM / MODEL SECURITY")

# Print a separator before starting Garak.
print("=" * 70)


# Execute Garak using the current Python environment.
result = subprocess.run(
    garak_command,

    # Pass the current environment to Garak.
    env=os.environ.copy(),
)


# Check whether Garak completed successfully.
if result.returncode != 0:

    # Display a clear failure status.
    print("\n" + "=" * 70)
    print("GARAK SCAN STATUS: FAILED")
    print("=" * 70)

    # Return Garak's original exit code.
    raise SystemExit(result.returncode)


# Display successful completion.
print("\n" + "=" * 70)
print("GARAK SCAN STATUS: COMPLETED")
print("=" * 70)


# Explain what Garak performed.
print("\nSECURITY TESTING FLOW")
print("-" * 70)

# Garak generated adversarial security tests.
print("1. Garak Probe")

# The probe creates adversarial prompts.
print("        ↓")

# The generated prompt is sent to the target model.
print("2. Adversarial Prompt")

# OpenAI processes the prompt.
print("        ↓")

# The target LLM generates a response.
print("3. OpenAI / gpt-5-mini")

# Garak analyzes the response.
print("        ↓")

# Garak detectors identify possible vulnerabilities.
print("4. Garak Detector")

# Garak records the security result.
print("        ↓")

# Garak writes the result to its JSONL report.
print("5. Garak JSONL Report")

print("-" * 70)

# Explain the scope of this test.
print("SCOPE")
print("-" * 70)

# This scan focuses on the target LLM itself.
print("This program tests LLM/model-level security.")

# It does not test application authorization or business logic.
print("It does NOT test application-level RBAC/tool authorization.")

# End the program.
print("=" * 70)