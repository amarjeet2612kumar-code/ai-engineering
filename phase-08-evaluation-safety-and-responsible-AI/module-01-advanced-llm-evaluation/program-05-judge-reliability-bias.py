import os
import json
import time
import requests

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load variables from the .env file.
load_dotenv()

# Define the Ollama API endpoint.
OLLAMA_URL = "http://localhost:11434/api/generate"

# Define the local model used to generate answers.
OLLAMA_MODEL = "llama3.2:3b"

# Read the Gemini API key from the environment.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Stop the program if the API key is missing.
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")

# Create the Gemini API client.
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Define the Gemini model used as the judge.
GEMINI_MODEL = "gemini-3.5-flash"


# ---------------------------------------------------------
# OLLAMA GENERATOR
# ---------------------------------------------------------

# Send a prompt to Ollama and return the generated answer.
def ask_ollama(prompt):

    # Prepare the Ollama request payload.
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

    # Raise an error if Ollama returns an error.
    response.raise_for_status()

    # Extract and return the generated answer.
    return response.json()["response"].strip()


# Generate an answer using Ollama.
def generate_answer(question, instruction):

    # Build the generator prompt.
    prompt = f"""
Answer the following question.

Additional instruction:
{instruction}

Question:
{question}
"""

    # Generate the answer using Ollama.
    return ask_ollama(prompt)


# ---------------------------------------------------------
# GEMINI JUDGE
# ---------------------------------------------------------

# Send an evaluation request to Gemini.
def call_gemini(prompt, schema):

    # Try the request up to three times.
    for attempt in range(1, 4):

        try:

            # Send the evaluation request to Gemini.
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema
                )
            )

            # Convert Gemini's JSON response into a Python dictionary.
            return json.loads(response.text)

        except Exception as error:

            # Display the failed attempt.
            print(
                f"Gemini attempt {attempt} failed: {error}"
            )

            # Wait before retrying.
            if attempt < 3:
                time.sleep(3)

    # Stop the program if all attempts fail.
    raise RuntimeError(
        "Gemini judge failed after 3 attempts."
    )


# ---------------------------------------------------------
# RELIABILITY EVALUATION
# ---------------------------------------------------------

# Evaluate the same answer multiple times.
def judge_answer(question, answer):

    # Define the evaluation prompt.
    prompt = f"""
You are an objective LLM evaluator.

Evaluate the answer using these criteria:

1. Correctness:
   Is the information factually correct?

2. Relevance:
   Does the answer directly answer the question?

3. Clarity:
   Is the answer easy to understand?

Give each score from 1 to 5.

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
                "type": "INTEGER"
            },
            "reason": {
                "type": "STRING"
            }
        },
        "required": [
            "correctness",
            "relevance",
            "clarity",
            "overall",
            "reason"
        ]
    }

    # Send the evaluation to Gemini.
    return call_gemini(prompt, schema)


# ---------------------------------------------------------
# PAIRWISE EVALUATION
# ---------------------------------------------------------

# Compare two answers using Gemini.
def pairwise_judge(question, answer_a, answer_b):

    # Define the pairwise evaluation prompt.
    prompt = f"""
You are an objective LLM evaluator.

Compare Answer A and Answer B for the question.

Choose the better answer.

Evaluate using:

1. Correctness
2. Relevance
3. Clarity

Do NOT prefer an answer simply because it appears first.

Choose exactly one winner:

A
B
TIE

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}
"""

    # Define the expected JSON structure.
    schema = {
        "type": "OBJECT",
        "properties": {
            "winner": {
                "type": "STRING"
            },
            "reason": {
                "type": "STRING"
            }
        },
        "required": [
            "winner",
            "reason"
        ]
    }

    # Send the comparison to Gemini.
    return call_gemini(prompt, schema)


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

# Define the question used for the experiment.
question = "What is Apache Spark?"


# =========================================================
# TEST 1: JUDGE RELIABILITY
# =========================================================

# Display the reliability test heading.
print("\n" + "=" * 60)
print("TEST 1 - JUDGE RELIABILITY")
print("=" * 60)


# Generate one answer from Ollama.
reliability_answer = generate_answer(
    question,
    "Give a concise and technically accurate definition."
)


# Display the answer that will be repeatedly judged.
print("\nAnswer being evaluated:")
print(reliability_answer)


# Store all judge results.
reliability_results = []


# Run the same evaluation three times.
for run in range(1, 4):

    # Display the current run.
    print(f"\n--- Judge Run {run} ---")

    # Ask Gemini to judge the same answer.
    result = judge_answer(
        question,
        reliability_answer
    )

    # Store the result.
    reliability_results.append(result)

    # Display the result.
    print(json.dumps(result, indent=4))


# =========================================================
# TEST 2: GENERATE ANSWER A
# =========================================================

# Display the Answer A heading.
print("\n" + "=" * 60)
print("TEST 2 - GENERATING ANSWER A")
print("=" * 60)


# Generate Answer A using a concise prompt.
answer_a = generate_answer(
    question,
    "Give a short, precise definition in 2-3 sentences."
)


# Display Answer A.
print("\nAnswer A:")
print(answer_a)


# =========================================================
# TEST 3: GENERATE ANSWER B
# =========================================================

# Display the Answer B heading.
print("\n" + "=" * 60)
print("TEST 3 - GENERATING ANSWER B")
print("=" * 60)


# Generate Answer B using a different prompt.
answer_b = generate_answer(
    question,
    """
Give a detailed explanation including features,
architecture, and common use cases.
"""
)


# Display Answer B.
print("\nAnswer B:")
print(answer_b)


# =========================================================
# TEST 4: POSITION BIAS
# =========================================================

# Display the position bias heading.
print("\n" + "=" * 60)
print("TEST 4 - POSITION BIAS")
print("=" * 60)


# Compare Answer A in the first position.
print("\n--- A vs B ---")

# Ask Gemini to compare A against B.
result_ab = pairwise_judge(
    question,
    answer_a,
    answer_b
)

# Display the result.
print(json.dumps(result_ab, indent=4))


# Compare Answer B in the first position.
print("\n--- B vs A ---")

# Ask Gemini to compare B against A.
result_ba = pairwise_judge(
    question,
    answer_b,
    answer_a
)

# Display the result.
print(json.dumps(result_ba, indent=4))


# =========================================================
# FINAL INTERPRETATION
# =========================================================

# Display the interpretation heading.
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)


# Explain the reliability experiment.
print("""
Reliability:
Compare the three judge runs.

If the scores are similar, the judge is more consistent.
If the scores vary significantly, the judge may have reliability issues.
""")


# Explain the position-bias experiment.
print("""
Position Bias:
Compare the winners from A vs B and B vs A.

If the same answer wins after its position changes,
the judge is less likely to have position bias.

If the winner always follows position A or position B,
the judge may have position bias.
""")