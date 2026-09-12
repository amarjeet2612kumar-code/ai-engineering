import requests  # Used to call the local Ollama API


# Store a small set of evaluation test cases.
# Each test case contains a question and the expected answer.
evaluation_dataset = [
    {
        "id": 1,
        "question": "What is Spark partitioning?",
        "reference_answer": "Spark partitioning divides data into smaller parts so they can be processed in parallel."
    },
    {
        "id": 2,
        "question": "What is a Kafka topic?",
        "reference_answer": "A Kafka topic is a logical category where Kafka stores and organizes messages."
    },
    {
        "id": 3,
        "question": "What is the purpose of an Airflow DAG?",
        "reference_answer": "An Airflow DAG defines the tasks and dependencies that make up a workflow."
    }
]


# Define the rubric used to manually evaluate each LLM response.
# Each criterion has a score from 1 to 5.
rubric = {
    "correctness": "Is the answer factually correct?",
    "relevance": "Does the answer directly answer the question?",
    "clarity": "Is the answer easy to understand?"
}


# Send a question to the local Ollama model.
def generate_answer(question):
    # Prepare the request payload for Ollama.
    payload = {
        "model": "llama3.2:3b",
        "prompt": question,
        "stream": False
    }

    # Call the Ollama generate API.
    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload
    )

    # Stop the program if Ollama returns an error.
    response.raise_for_status()

    # Return only the generated answer.
    return response.json()["response"].strip()


# Display the rubric so we know how each response should be scored.
def show_rubric():
    print("\nEvaluation Rubric")
    print("-----------------")

    # Print every evaluation criterion.
    for criterion, description in rubric.items():
        print(f"{criterion.capitalize()}: {description}")

    # Explain the scoring scale.
    print("\nScore: 1 = Poor, 5 = Excellent")


# Ask the evaluator to manually score one response.
def get_score(criterion):
    # Keep asking until the evaluator enters a valid score.
    while True:
        try:
            score = int(input(f"{criterion.capitalize()} score (1-5): "))

            # Accept only scores between 1 and 5.
            if 1 <= score <= 5:
                return score

            # Inform the evaluator about the valid range.
            print("Please enter a score between 1 and 5.")

        except ValueError:
            # Handle non-numeric input.
            print("Please enter a number between 1 and 5.")


# Evaluate every test case in the dataset.
def evaluate_dataset():
    # Display the rubric before starting evaluation.
    show_rubric()

    # Process each test case one by one.
    for test_case in evaluation_dataset:
        print("\n" + "=" * 60)
        print(f"Test Case: {test_case['id']}")
        print(f"Question: {test_case['question']}")

        # Generate an answer using the local LLM.
        answer = generate_answer(test_case["question"])

        # Display the trusted reference answer.
        print(f"\nReference Answer:\n{test_case['reference_answer']}")

        # Display the generated LLM answer.
        print(f"\nLLM Answer:\n{answer}")

        print("\nEnter your evaluation scores:")

        # Collect a score for each rubric criterion.
        scores = {}

        for criterion in rubric:
            scores[criterion] = get_score(criterion)

        # Calculate the average score across all criteria.
        total_score = sum(scores.values())
        average_score = total_score / len(scores)

        # Display the evaluation result.
        print("\nEvaluation Result")
        print("-----------------")
        print(f"Correctness: {scores['correctness']}/5")
        print(f"Relevance:   {scores['relevance']}/5")
        print(f"Clarity:     {scores['clarity']}/5")
        print(f"Average:     {average_score:.2f}/5")


# Start the evaluation program.
if __name__ == "__main__":
    evaluate_dataset()