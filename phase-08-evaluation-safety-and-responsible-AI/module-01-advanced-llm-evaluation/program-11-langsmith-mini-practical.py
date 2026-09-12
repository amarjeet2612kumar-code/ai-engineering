# Import os to access environment variables.
import os

# Import load_dotenv to load variables from the .env file.
from dotenv import load_dotenv

# Import the OpenAI client.
from openai import OpenAI

# Import LangSmith client for creating traces.
from langsmith import Client


# Load environment variables from the .env file.
load_dotenv()


# Read the OpenAI API key.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Read the LangSmith API key.
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")


# Read the LangSmith tracing setting.
LANGSMITH_TRACING = os.getenv(
    "LANGSMITH_TRACING",
    "false",
)


# Read the LangSmith project name.
LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT",
    "observability-test",
)


# Stop the program if the OpenAI API key is missing.
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set. "
        "Check your .env file."
    )


# Define the OpenAI model used for generation.
MODEL_NAME = "gpt-5-mini"


# Create the OpenAI client.
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
)


# Create the LangSmith client only when an API key exists.
langsmith_client = None

# Check whether a LangSmith API key is configured.
if LANGSMITH_API_KEY:

    # Create the LangSmith client.
    langsmith_client = Client(
        api_key=LANGSMITH_API_KEY,
    )


# Define a small set of questions.
evaluation_dataset = [
    {
        "question": "What is Apache Spark?",
    },
    {
        "question": "What is Apache Kafka?",
    },
    {
        "question": "What is Apache Airflow?",
    },
]


# Generate an answer and optionally send a trace to LangSmith.
def generate_answer(question):

    # Create a LangSmith run only when authentication is available.
    run = None

    # Check whether LangSmith is configured.
    if langsmith_client:

        # Create an LLM trace in the configured LangSmith project.
        run = langsmith_client.create_run(
            name="llm-generation",
            run_type="llm",
            inputs={
                "question": question,
            },
            project_name=LANGSMITH_PROJECT,
        )

    try:

        # Send the question to OpenAI.
        response = openai_client.responses.create(
            model=MODEL_NAME,
            input=question,
        )

        # Extract the generated answer.
        answer = response.output_text

        # Send the generated answer to LangSmith when tracing is enabled.
        if run:

            # Update the LangSmith run with the generated output.
            langsmith_client.update_run(
                run.id,
                outputs={
                    "answer": answer,
                },
            )

        # Return the generated answer.
        return answer

    except Exception as error:

        # Record the error in LangSmith when a run exists.
        if run:

            # Update the LangSmith run with the error.
            langsmith_client.update_run(
                run.id,
                error=str(error),
            )

        # Re-raise the error.
        raise


# Run the practical.
def main():

    # Print the program header.
    print("=" * 70)
    print("LANGSMITH MINI PRACTICAL")
    print("=" * 70)

    # Display the OpenAI model.
    print(f"Model: {MODEL_NAME}")

    # Display the LangSmith project.
    print(f"LangSmith Project: {LANGSMITH_PROJECT}")

    # Display whether a LangSmith API key exists.
    print(
        f"LangSmith API Key Configured: "
        f"{bool(LANGSMITH_API_KEY)}"
    )

    # Display the tracing configuration.
    print(
        f"LangSmith Tracing Setting: "
        f"{LANGSMITH_TRACING}"
    )

    # Print a separator.
    print("=" * 70)


    # Process each question.
    for index, item in enumerate(
        evaluation_dataset,
        start=1,
    ):

        # Extract the current question.
        question = item["question"]

        # Display the current question.
        print(f"\nQuestion {index}: {question}")

        # Generate the answer.
        answer = generate_answer(question)

        # Display the generated answer.
        print(f"Answer: {answer}")


    # Print the completion section.
    print("\n" + "=" * 70)
    print("PRACTICAL COMPLETED")
    print("=" * 70)


    # Explain the current LangSmith status.
    if langsmith_client:

        # Confirm that traces were sent to LangSmith.
        print(
            "LangSmith tracing is configured."
        )

        # Display the project where traces were sent.
        print(
            f"Project: {LANGSMITH_PROJECT}"
        )

    else:

        # Explain that no LangSmith API key was configured.
        print(
            "LangSmith API key is empty."
        )

        # Explain that the OpenAI generation still worked.
        print(
            "OpenAI generation completed, "
            "but no LangSmith traces were sent."
        )


# Start the program.
if __name__ == "__main__":
    main()