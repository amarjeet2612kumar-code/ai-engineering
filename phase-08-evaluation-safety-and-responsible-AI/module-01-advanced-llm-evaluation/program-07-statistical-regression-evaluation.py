import os
import json
import time
import statistics
import requests

from dotenv import load_dotenv
from groq import Groq
from google import genai
from google.genai import types


# Load variables from the .env file.
load_dotenv()


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Read the Ollama host from the environment.
OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
).rstrip("/")

# Build the Ollama API URL.
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"

# Define the Ollama model.
OLLAMA_MODEL = "llama3.2:3b"

# Define the Groq model.
GROQ_MODEL = "qwen/qwen3.6-27b"

# Define the Gemini judge model.
GEMINI_MODEL = "gemini-3.5-flash"

# Read the Groq API key.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Read the Gemini API key.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Validate the Groq API key.
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env")

# Validate the Gemini API key.
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")


# Create the Groq client.
groq_client = Groq(api_key=GROQ_API_KEY)

# Create the Gemini client.
gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ---------------------------------------------------------
# VERSION A — OLLAMA
# ---------------------------------------------------------

# Generate an answer using Version A.
def generate_version_a(question):

    # Define the prompt for Version A.
    prompt = f"""
Answer the following question clearly and accurately.

Keep the answer under 300 words.

Question:
{question}
"""

    # Prepare the Ollama request.
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    # Send the request to Ollama.
    response = requests.post(
        OLLAMA_URL,
        json=payload
    )

    # Raise an error if the request fails.
    response.raise_for_status()

    # Return the generated answer.
    return response.json()["response"].strip()


# ---------------------------------------------------------
# VERSION B — GROQ
# ---------------------------------------------------------

# Generate an answer using Version B.
def generate_version_b(question):

    # Send the question to Groq.
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""
Answer the following question clearly and accurately.

Keep the answer under 300 words.

Question:
{question}
"""
            }
        ],

        # Use deterministic generation.
        temperature=0,

        # Limit output tokens.
        max_tokens=600
    )

    # Return the generated answer.
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------
# GEMINI JUDGE
# ---------------------------------------------------------

# Ask Gemini to score one answer.
def judge_answer(question, answer):

    # Define the evaluation prompt.
    prompt = f"""
You are an objective evaluator.

Evaluate the answer using:

1. Correctness
2. Relevance
3. Clarity

Give each criterion a score from 1 to 5.

Calculate the overall score as the average
of correctness, relevance, and clarity.

Question:
{question}

Answer:
{answer}
"""

    # Define the expected JSON structure.
    schema = {
        "type": "OBJECT",
        "properties": {
            "correctness": {
                "type": "INTEGER"
            },
            "relevance": {
                "type": "INTEGER"
            },
            "clarity": {
                "type": "INTEGER"
            },
            "overall": {
                "type": "NUMBER"
            }
        },
        "required": [
            "correctness",
            "relevance",
            "clarity",
            "overall"
        ]
    }

    # Retry Gemini if the service is temporarily unavailable.
    for attempt in range(1, 4):

        try:

            # Send the evaluation request.
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema
                )
            )

            # Convert the JSON response into a dictionary.
            return json.loads(response.text)

        except Exception as error:

            # Display the failed attempt.
            print(
                f"Gemini attempt {attempt} failed: {error}"
            )

            # Wait before retrying.
            if attempt < 3:
                time.sleep(3)

    # Stop after all retries fail.
    raise RuntimeError(
        "Gemini judge failed after 3 attempts."
    )


# ---------------------------------------------------------
# EVALUATION DATASET
# ---------------------------------------------------------

# Define the common evaluation dataset.
evaluation_cases = [
    "What is Apache Spark?",
    "What is Apache Kafka?",
    "What is Apache Airflow?",
    "What is a Kafka topic?",
    "What is a Spark DataFrame?",
    "What is an Airflow DAG?"
]


# Store Version A results.
version_a_scores = []

# Store Version B results.
version_b_scores = []


# ---------------------------------------------------------
# RUN EVALUATION
# ---------------------------------------------------------

# Process every evaluation question.
for index, question in enumerate(
    evaluation_cases,
    start=1
):

    # Display the test case.
    print("\n" + "=" * 70)
    print(f"TEST CASE {index}")
    print("=" * 70)

    # Display the question.
    print(f"\nQuestion:\n{question}")


    # -----------------------------------------------------
    # VERSION A
    # -----------------------------------------------------

    # Generate the Version A answer.
    answer_a = generate_version_a(question)

    # Display the Version A answer.
    print("\n--- VERSION A — OLLAMA ---")
    print(answer_a)

    # Evaluate Version A.
    score_a = judge_answer(
        question,
        answer_a
    )

    # Store the Version A overall score.
    version_a_scores.append(
        score_a["overall"]
    )

    # Display the Version A score.
    print("\nVersion A Score:")
    print(json.dumps(score_a, indent=4))


    # -----------------------------------------------------
    # VERSION B
    # -----------------------------------------------------

    # Generate the Version B answer.
    answer_b = generate_version_b(question)

    # Display the Version B answer.
    print("\n--- VERSION B — GROQ ---")
    print(answer_b)

    # Evaluate Version B.
    score_b = judge_answer(
        question,
        answer_b
    )

    # Store the Version B overall score.
    version_b_scores.append(
        score_b["overall"]
    )

    # Display the Version B score.
    print("\nVersion B Score:")
    print(json.dumps(score_b, indent=4))


# ---------------------------------------------------------
# STATISTICAL ANALYSIS
# ---------------------------------------------------------

# Calculate the average score for Version A.
mean_a = statistics.mean(version_a_scores)

# Calculate the average score for Version B.
mean_b = statistics.mean(version_b_scores)

# Calculate the standard deviation for Version A.
std_a = statistics.stdev(version_a_scores)

# Calculate the standard deviation for Version B.
std_b = statistics.stdev(version_b_scores)

# Calculate the minimum Version A score.
min_a = min(version_a_scores)

# Calculate the minimum Version B score.
min_b = min(version_b_scores)

# Calculate the maximum Version A score.
max_a = max(version_a_scores)

# Calculate the maximum Version B score.
max_b = max(version_b_scores)

# Calculate the change between versions.
difference = mean_b - mean_a

# Calculate the percentage change.
percentage_change = (
    (difference / mean_a) * 100
    if mean_a != 0
    else 0
)


# ---------------------------------------------------------
# DISPLAY STATISTICS
# ---------------------------------------------------------

# Display the statistical analysis heading.
print("\n" + "=" * 70)
print("STATISTICAL EVALUATION")
print("=" * 70)


# Display Version A scores.
print(f"\nVersion A Scores: {version_a_scores}")

# Display Version B scores.
print(f"Version B Scores: {version_b_scores}")

# Display Version A statistics.
print("\nVersion A:")
print(f"Mean: {mean_a:.2f}")
print(f"Std Dev: {std_a:.2f}")
print(f"Min: {min_a:.2f}")
print(f"Max: {max_a:.2f}")

# Display Version B statistics.
print("\nVersion B:")
print(f"Mean: {mean_b:.2f}")
print(f"Std Dev: {std_b:.2f}")
print(f"Min: {min_b:.2f}")
print(f"Max: {max_b:.2f}")

# Display the difference between versions.
print(f"\nMean Difference: {difference:.2f}")

# Display the percentage change.
print(
    f"Percentage Change: {percentage_change:.2f}%"
)


# ---------------------------------------------------------
# REGRESSION CHECK
# ---------------------------------------------------------

# Display the regression heading.
print("\n" + "=" * 70)
print("REGRESSION CHECK")
print("=" * 70)


# Check whether the new version is worse overall.
if mean_b < mean_a:

    # Calculate the regression amount.
    regression = mean_a - mean_b

    # Display the regression result.
    print(
        f"\nREGRESSION DETECTED: "
        f"Version B is {regression:.2f} points worse."
    )

else:

    # Display the improvement result.
    print(
        "\nNO OVERALL REGRESSION: "
        "Version B performs equal to or better than Version A."
    )