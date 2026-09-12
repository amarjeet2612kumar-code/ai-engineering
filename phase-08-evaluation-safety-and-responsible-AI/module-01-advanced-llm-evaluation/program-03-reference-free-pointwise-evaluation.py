import requests

# Define the Ollama API endpoint.
OLLAMA_URL = "http://localhost:11434/api/generate"

# Define the local LLM model.
MODEL = "llama3.2:3b"


# Send a question to Ollama and return the generated answer.
def ask_ollama(question):
    # Prepare the request payload.
    payload = {
        "model": MODEL,
        "prompt": question,
        "stream": False
    }

    # Send the request to Ollama.
    response = requests.post(OLLAMA_URL, json=payload)

    # Raise an error if the request failed.
    response.raise_for_status()

    # Return the generated answer.
    return response.json()["response"].strip()


# Ask the human evaluator for a score between 1 and 5.
def get_score(criteria):
    # Keep asking until the evaluator provides a valid score.
    while True:
        try:
            # Ask the evaluator to enter a score.
            score = int(input(f"{criteria} score (1-5): "))

            # Validate that the score is between 1 and 5.
            if 1 <= score <= 5:
                return score

            # Inform the evaluator about the valid range.
            print("Please enter a score between 1 and 5.")

        except ValueError:
            # Handle non-numeric input.
            print("Please enter a number between 1 and 5.")


# Define evaluation questions.
evaluation_cases = [
    "What is Apache Spark?",
    "What is Apache Kafka?",
    "What is Apache Airflow?"
]


# Evaluate each generated response independently.
for index, question in enumerate(evaluation_cases, start=1):

    # Display the test case number.
    print(f"\n{'=' * 60}")
    print(f"TEST CASE {index}")
    print(f"{'=' * 60}")

    # Display the question.
    print(f"\nQuestion:\n{question}")

    # Generate an answer using Ollama.
    generated_answer = ask_ollama(question)

    # Display the generated answer.
    print(f"\nGenerated Answer:\n{generated_answer}")

    # Explain the evaluation criteria.
    print("\nEvaluate the answer using these criteria:")
    print("1. Correctness - Is the information factually correct?")
    print("2. Relevance   - Does it directly answer the question?")
    print("3. Clarity     - Is it easy to understand?")

    # Collect the correctness score.
    correctness = get_score("Correctness")

    # Collect the relevance score.
    relevance = get_score("Relevance")

    # Collect the clarity score.
    clarity = get_score("Clarity")

    # Calculate the average pointwise score.
    average = (correctness + relevance + clarity) / 3

    # Display the evaluation result.
    print(f"\nPointwise Evaluation Score: {average:.2f}/5")