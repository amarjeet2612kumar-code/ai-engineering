import os  # Provides access to environment variables.

from dotenv import load_dotenv  # Loads variables from the .env file.

from deepeval import evaluate  # Runs DeepEval evaluations.

from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
)  # Defines test cases and valid evaluation parameters.

from deepeval.metrics import GEval  # Provides LLM-as-a-judge custom evaluation.

from deepeval.models import OllamaModel  # Allows DeepEval to use Ollama.


# ======================================================================
# ENVIRONMENT
# ======================================================================

load_dotenv()  # Load variables from the .env file.

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)  # Read the Ollama server address.

OLLAMA_MODEL = "llama3.2:3b"  # Use the local Ollama model.


# ======================================================================
# DEEPEVAL JUDGE MODEL
# ======================================================================

judge_model = OllamaModel(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_HOST,
)  # Configure Ollama as the DeepEval judge.


# ======================================================================
# CORRECTNESS METRIC
# ======================================================================

correctness_metric = GEval(
    name="Correctness",  # Give the metric a readable name.

    criteria=(
        "Evaluate whether the actual answer is factually correct "
        "and accurately answers the question."
    ),  # Define the correctness evaluation criteria.

    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],  # Specify the LLMTestCase fields used by this metric.

    model=judge_model,  # Use Ollama as the judge.

    threshold=0.7,  # Require a minimum score of 0.7.
)


# ======================================================================
# RELEVANCE METRIC
# ======================================================================

relevance_metric = GEval(
    name="Relevance",  # Give the metric a readable name.

    criteria=(
        "Evaluate whether the answer directly addresses "
        "the question without unnecessary information."
    ),  # Define the relevance evaluation criteria.

    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],  # Specify the LLMTestCase fields used by this metric.

    model=judge_model,  # Use Ollama as the judge.

    threshold=0.7,  # Require a minimum score of 0.7.
)


# ======================================================================
# EVALUATION DATASET
# ======================================================================

test_cases = [

    LLMTestCase(
        input="What is Apache Spark?",  # Define the evaluation question.

        actual_output=(
            "Apache Spark is an open-source distributed "
            "data processing engine used for large-scale "
            "data processing."
        ),  # Define the answer being evaluated.
    ),

    LLMTestCase(
        input="What is Apache Kafka?",  # Define the evaluation question.

        actual_output=(
            "Apache Kafka is a distributed event streaming "
            "platform used to publish, store, and consume "
            "streams of records."
        ),  # Define the answer being evaluated.
    ),

    LLMTestCase(
        input="What is Apache Airflow?",  # Define the evaluation question.

        actual_output=(
            "Apache Airflow is an open-source platform "
            "used to programmatically author, schedule, "
            "and monitor workflows."
        ),  # Define the answer being evaluated.
    ),
]


# ======================================================================
# RUN EVALUATION
# ======================================================================

print("=" * 70)  # Print the program header.

print("DEEPEVAL PRACTICAL")  # Display the program title.

print("=" * 70)  # Print the separator.

print(
    f"Judge Model: {OLLAMA_MODEL}"
)  # Display the judge model.

print(
    "Metrics: Correctness + Relevance"
)  # Display the metrics being evaluated.


# Run the test cases against both DeepEval metrics.
results = evaluate(
    test_cases=test_cases,
    metrics=[
        correctness_metric,
        relevance_metric,
    ],
)


# ======================================================================
# COMPLETION
# ======================================================================

print("\n" + "=" * 70)  # Print the final separator.

print("DEEPEVAL EVALUATION COMPLETED")  # Confirm evaluation completion.

print("=" * 70)  # Print the final separator.