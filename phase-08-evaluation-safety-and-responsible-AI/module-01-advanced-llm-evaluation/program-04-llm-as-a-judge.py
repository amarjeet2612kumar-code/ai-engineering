import requests
import json

# Define the Ollama API endpoint.
OLLAMA_URL = "http://localhost:11434/api/generate"

# Define the local model used for both generation and judging.
MODEL = "llama3.2:3b"


# Send a prompt to Ollama and return the generated response.
def ask_ollama(prompt):
    # Prepare the Ollama request.
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    # Send the request to Ollama.
    response = requests.post(OLLAMA_URL, json=payload)

    # Raise an error if the request failed.
    response.raise_for_status()

    # Return the generated response.
    return response.json()["response"].strip()


# Generate an answer for the user's question.
def generate_answer(question):
    # Ask the generator LLM to answer the question.
    prompt = f"""
Answer the following question clearly and accurately.

Question:
{question}
"""

    # Return the generated answer.
    return ask_ollama(prompt)


# Ask the judge LLM to evaluate the generated answer.
def judge_answer(question, answer):
    # Define the judge's evaluation instructions.
    prompt = f"""
You are an evaluator.

Evaluate the answer using these criteria:

1. Correctness: Is the information factually correct?
2. Relevance: Does the answer directly address the question?
3. Clarity: Is the answer easy to understand?

Give each criterion a score from 1 to 5.

Return ONLY valid JSON in this format:

{{
    "correctness": 1,
    "relevance": 1,
    "clarity": 1,
    "overall": 1,
    "reason": "short explanation"
}}

Question:
{question}

Answer:
{answer}
"""

    # Ask the LLM to evaluate the answer.
    judge_response = ask_ollama(prompt)

    # Return the raw judge response.
    return judge_response


# Define questions for evaluation.
evaluation_cases = [
    "What is Apache Spark?",
    "What is Apache Kafka?",
    "What is Apache Airflow?"
]


# Evaluate each question.
for index, question in enumerate(evaluation_cases, start=1):

    # Display the test case number.
    print(f"\n{'=' * 60}")
    print(f"TEST CASE {index}")
    print(f"{'=' * 60}")

    # Display the question.
    print(f"\nQuestion:\n{question}")

    # Generate the answer using the generator LLM.
    answer = generate_answer(question)

    # Display the generated answer.
    print(f"\nGenerated Answer:\n{answer}")

    # Ask the judge LLM to evaluate the generated answer.
    judge_result = judge_answer(question, answer)

    # Display the judge's evaluation.
    print(f"\nJudge Evaluation:\n{judge_result}")