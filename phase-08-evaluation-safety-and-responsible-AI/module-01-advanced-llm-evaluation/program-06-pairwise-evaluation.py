import os
import json
import time
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

# Build the Ollama generate API URL.
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"

# Define the Ollama model used as Generator A.
OLLAMA_MODEL = "llama3.2:3b"

# Read the Groq API key from the environment.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Read the Gemini API key from the environment.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Stop the program when the Groq key is missing.
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env")

# Stop the program when the Gemini key is missing.
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")


# Create the Groq client.
groq_client = Groq(api_key=GROQ_API_KEY)

# Create the Gemini client.
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# Define the Groq model used as Generator B.
GROQ_MODEL = "qwen/qwen3.6-27b"

# Define the Gemini model used as the independent judge.
GEMINI_MODEL = "gemini-3.5-flash"


# ---------------------------------------------------------
# GENERATOR A — OLLAMA
# ---------------------------------------------------------

# Generate an answer using the local Ollama model.
def generate_with_ollama(question):

    # Define the prompt given to Ollama.
    prompt = f"""
Answer the following question clearly and accurately.

Keep the answer under 400 words.

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

    # Raise an exception if Ollama returns an error.
    response.raise_for_status()

    # Extract the generated answer.
    return response.json()["response"].strip()


# ---------------------------------------------------------
# GENERATOR B — GROQ
# ---------------------------------------------------------

# Generate an answer using the Groq model.
def generate_with_groq(question):

    # Send the question to the Groq model.
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""
Answer the following question clearly and accurately.

Keep the answer under 400 words.

Question:
{question}
"""
            }
        ],

        # Use deterministic generation for easier comparison.
        temperature=0,

        # Limit output tokens to stay within the available Groq limit.
        max_tokens=700
    )

    # Extract and return the generated answer.
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------
# GEMINI JUDGE
# ---------------------------------------------------------

# Ask Gemini to compare two generated answers.
def judge_pair(question, answer_a, answer_b):

    # Define the pairwise evaluation prompt.
    prompt = f"""
You are an independent and objective LLM evaluator.

Compare Answer A and Answer B for the question.

Choose the better answer based on:

1. Correctness
2. Relevance
3. Clarity
4. Technical accuracy

Do not prefer an answer because:
- It is longer.
- It is shorter.
- It appears first.
- It uses more sophisticated language.

Choose exactly one winner:
A
B
TIE

Explain briefly why the selected answer is better.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}
"""

    # Define the JSON structure expected from Gemini.
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

    # Try the Gemini request up to three times.
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

            # Convert Gemini's JSON response to a Python dictionary.
            return json.loads(response.text)

        except Exception as error:

            # Display the failed attempt.
            print(
                f"\nGemini attempt {attempt} failed: {error}"
            )

            # Wait before retrying.
            if attempt < 3:
                time.sleep(3)

    # Stop the program if Gemini remains unavailable.
    raise RuntimeError(
        "Gemini judge failed after 3 attempts."
    )


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

# Define the questions used for pairwise evaluation.
evaluation_cases = [
    "What is Apache Spark?",
    "What is Apache Kafka?",
    "What is Apache Airflow?"
]


# Process each evaluation question.
for index, question in enumerate(evaluation_cases, start=1):

    # Display the test case heading.
    print("\n" + "=" * 70)
    print(f"TEST CASE {index}")
    print("=" * 70)

    # Display the question.
    print(f"\nQuestion:\n{question}")


    # -----------------------------------------------------
    # GENERATE ANSWER A
    # -----------------------------------------------------

    # Generate Answer A using Ollama.
    answer_a = generate_with_ollama(question)

    # Display Answer A.
    print("\n" + "-" * 70)
    print("ANSWER A — OLLAMA")
    print("-" * 70)
    print(answer_a)


    # -----------------------------------------------------
    # GENERATE ANSWER B
    # -----------------------------------------------------

    # Generate Answer B using Groq.
    answer_b = generate_with_groq(question)

    # Display Answer B.
    print("\n" + "-" * 70)
    print("ANSWER B — GROQ")
    print("-" * 70)
    print(answer_b)


    # -----------------------------------------------------
    # GEMINI PAIRWISE JUDGE
    # -----------------------------------------------------

    # Ask Gemini to compare Answer A and Answer B.
    result = judge_pair(
        question,
        answer_a,
        answer_b
    )

    # Display the judge result.
    print("\n" + "-" * 70)
    print("GEMINI PAIRWISE JUDGE")
    print("-" * 70)

    # Display the structured JSON evaluation.
    print(json.dumps(result, indent=4))