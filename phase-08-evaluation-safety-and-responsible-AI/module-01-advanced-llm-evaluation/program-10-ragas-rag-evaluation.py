# Import os to access environment variables.
import os

# Import asyncio for asynchronous Ragas evaluation.
import asyncio

# Import pandas to display evaluation results.
import pandas as pd

# Import load_dotenv to load variables from the .env file.
from dotenv import load_dotenv

# Import AsyncOpenAI for asynchronous OpenAI API access.
from openai import AsyncOpenAI

# Import Ragas LLM factory for modern OpenAI integration.
from ragas.llms import llm_factory

# Import Ragas embedding factory for AnswerRelevancy.
from ragas.embeddings.base import embedding_factory

# Import modern Ragas 0.4.x metric implementations.
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)


# Load variables from the project .env file.
load_dotenv()


# Define the OpenAI model used as the Ragas judge.
OPENAI_MODEL = "gpt-5-mini"


# Read the OpenAI API key from the environment.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Stop execution if the API key cannot be found.
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set. "
        "Check that your .env file contains OPENAI_API_KEY."
    )


# Create an asynchronous OpenAI client.
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    timeout=60.0,
    max_retries=3,
)


# Create the Ragas LLM using the OpenAI client.
ragas_llm = llm_factory(
    OPENAI_MODEL,
    client=openai_client,
)


# Create the OpenAI embedding model used by AnswerRelevancy.
ragas_embeddings = embedding_factory(
    "openai",
    model="text-embedding-3-small",
    client=openai_client,
)


# Create the Faithfulness metric.
faithfulness_metric = Faithfulness(
    llm=ragas_llm,
)


# Create the AnswerRelevancy metric.
answer_relevancy_metric = AnswerRelevancy(
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)


# Create the ContextPrecision metric.
context_precision_metric = ContextPrecision(
    llm=ragas_llm,
)


# Create the ContextRecall metric.
context_recall_metric = ContextRecall(
    llm=ragas_llm,
)


# Define a small RAG evaluation dataset.
evaluation_dataset = [
    {
        "user_input": "What is Apache Spark?",
        "retrieved_contexts": [
            (
                "Apache Spark is an open-source distributed data processing "
                "engine designed for large-scale data processing."
            )
        ],
        "response": (
            "Apache Spark is an open-source distributed data processing "
            "engine used for large-scale data processing."
        ),
        "reference": (
            "Apache Spark is an open-source distributed data processing "
            "engine designed for large-scale data processing."
        ),
    },
    {
        "user_input": "What is Apache Kafka?",
        "retrieved_contexts": [
            (
                "Apache Kafka is a distributed event streaming platform "
                "used to publish, store, and consume streams of records."
            )
        ],
        "response": (
            "Apache Kafka is a distributed event streaming platform used "
            "to publish, store, and consume streams of records."
        ),
        "reference": (
            "Apache Kafka is a distributed event streaming platform used "
            "to publish, store, and consume streams of records."
        ),
    },
    {
        "user_input": "What is Apache Airflow?",
        "retrieved_contexts": [
            (
                "Apache Airflow is an open-source platform for "
                "programmatically authoring, scheduling, and monitoring "
                "workflows."
            )
        ],
        "response": (
            "Apache Airflow is an open-source platform used to "
            "programmatically author, schedule, and monitor workflows."
        ),
        "reference": (
            "Apache Airflow is an open-source platform for "
            "programmatically authoring, scheduling, and monitoring "
            "workflows."
        ),
    },
]


# Evaluate one RAG test case.
async def evaluate_case(case):

    # Evaluate whether the generated answer is supported by the retrieved context.
    faithfulness_result = await faithfulness_metric.ascore(
        user_input=case["user_input"],
        response=case["response"],
        retrieved_contexts=case["retrieved_contexts"],
    )

    # Evaluate whether the generated answer is relevant to the question.
    answer_relevancy_result = await answer_relevancy_metric.ascore(
        user_input=case["user_input"],
        response=case["response"],
    )

    # Evaluate whether the retrieved context is relevant to the reference answer.
    # ContextPrecisionWithReference does not accept the generated response.
    context_precision_result = await context_precision_metric.ascore(
        user_input=case["user_input"],
        retrieved_contexts=case["retrieved_contexts"],
        reference=case["reference"],
    )

    # Evaluate whether the retrieved context contains the information needed
    # to answer the reference question.
    context_recall_result = await context_recall_metric.ascore(
        user_input=case["user_input"],
        retrieved_contexts=case["retrieved_contexts"],
        reference=case["reference"],
    )

    # Return all four Ragas scores.
    return {
        "user_input": case["user_input"],
        "response": case["response"],
        "faithfulness": faithfulness_result.value,
        "answer_relevancy": answer_relevancy_result.value,
        "context_precision": context_precision_result.value,
        "context_recall": context_recall_result.value,
    }


# Run the complete Ragas evaluation.
async def main():

    # Print the program header.
    print("=" * 70)
    print("RAGAS RAG EVALUATION")
    print("=" * 70)

    # Display the judge model.
    print(f"Judge Model: {OPENAI_MODEL}")

    # Display the metrics being evaluated.
    print(
        "Metrics: Faithfulness + Answer Relevancy + "
        "Context Precision + Context Recall"
    )

    # Display the number of evaluation cases.
    print(f"Evaluation Cases: {len(evaluation_dataset)}")

    # Print a separator.
    print("=" * 70)

    # Store successful evaluation results.
    results = []

    # Evaluate each test case sequentially.
    for index, case in enumerate(evaluation_dataset, start=1):

        # Display the current test case.
        print(
            f"\nEvaluating Case {index}/{len(evaluation_dataset)}: "
            f"{case['user_input']}"
        )

        try:

            # Run all Ragas metrics for the current case.
            result = await evaluate_case(case)

            # Store the result.
            results.append(result)

            # Display Faithfulness.
            print(
                f"  Faithfulness:       "
                f"{result['faithfulness']:.4f}"
            )

            # Display Answer Relevancy.
            print(
                f"  Answer Relevancy:   "
                f"{result['answer_relevancy']:.4f}"
            )

            # Display Context Precision.
            print(
                f"  Context Precision:  "
                f"{result['context_precision']:.4f}"
            )

            # Display Context Recall.
            print(
                f"  Context Recall:     "
                f"{result['context_recall']:.4f}"
            )

        except Exception as error:

            # Print the error without exposing the API key.
            print(
                f"  Evaluation failed: "
                f"{type(error).__name__}: {error}"
            )


    # Stop if every evaluation failed.
    if not results:
        raise RuntimeError(
            "No evaluation case completed successfully."
        )


    # Convert successful results into a DataFrame.
    results_df = pd.DataFrame(results)


    # Calculate average Faithfulness.
    avg_faithfulness = results_df["faithfulness"].mean()

    # Calculate average Answer Relevancy.
    avg_answer_relevancy = results_df["answer_relevancy"].mean()

    # Calculate average Context Precision.
    avg_context_precision = results_df["context_precision"].mean()

    # Calculate average Context Recall.
    avg_context_recall = results_df["context_recall"].mean()


    # Print final results.
    print("\n" + "=" * 70)
    print("RAGAS RESULTS")
    print("=" * 70)

    # Display average Faithfulness.
    print(
        f"Average Faithfulness:      "
        f"{avg_faithfulness:.4f}"
    )

    # Display average Answer Relevancy.
    print(
        f"Average Answer Relevancy:  "
        f"{avg_answer_relevancy:.4f}"
    )

    # Display average Context Precision.
    print(
        f"Average Context Precision: "
        f"{avg_context_precision:.4f}"
    )

    # Display average Context Recall.
    print(
        f"Average Context Recall:    "
        f"{avg_context_recall:.4f}"
    )


    # Print detailed results.
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)

    # Select the important evaluation columns.
    display_columns = [
        "user_input",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    # Display per-case scores.
    print(
        results_df[display_columns].to_string(
            index=False
        )
    )


    # Print metric explanations.
    print("\n" + "=" * 70)
    print("METRIC INTERPRETATION")
    print("=" * 70)

    # Explain Faithfulness.
    print(
        "Faithfulness       -> "
        "Is the answer supported by retrieved context?"
    )

    # Explain Answer Relevancy.
    print(
        "Answer Relevancy   -> "
        "Does the answer address the user's question?"
    )

    # Explain Context Precision.
    print(
        "Context Precision  -> "
        "Are the retrieved contexts relevant?"
    )

    # Explain Context Recall.
    print(
        "Context Recall     -> "
        "Did retrieval contain the required information?"
    )


# Start the asynchronous program.
if __name__ == "__main__":
    asyncio.run(main())