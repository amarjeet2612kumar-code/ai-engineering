import requests
from sentence_transformers import SentenceTransformer, util

# Define the Ollama API endpoint.
OLLAMA_URL = "http://localhost:11434/api/generate"

# Define the local LLM model.
MODEL = "llama3.2:3b"

# Load the sentence-transformer model for semantic similarity.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Send a question to Ollama and return the generated answer.
def ask_ollama(question):
    # Prepare the request payload.
    payload = {
        "model": MODEL,
        "prompt": question,
        "stream": False
    }

    # Send the request to the Ollama API.
    response = requests.post(OLLAMA_URL, json=payload)

    # Raise an error if the API request failed.
    response.raise_for_status()

    # Extract the generated answer from the response.
    return response.json()["response"].strip()


# Compare two answers using exact string matching.
def exact_match(generated, reference):
    # Normalize both answers before comparison.
    generated = generated.strip().lower()
    reference = reference.strip().lower()

    # Return 1 when both answers are exactly the same.
    return 1.0 if generated == reference else 0.0


# Compare two answers based on their semantic meaning.
def semantic_similarity(generated, reference):
    # Convert the generated answer into an embedding.
    generated_embedding = embedding_model.encode(
        generated,
        convert_to_tensor=True
    )

    # Convert the reference answer into an embedding.
    reference_embedding = embedding_model.encode(
        reference,
        convert_to_tensor=True
    )

    # Calculate cosine similarity between the two embeddings.
    score = util.cos_sim(
        generated_embedding,
        reference_embedding
    ).item()

    # Return the similarity score.
    return score


# Define evaluation cases with trusted reference answers.
evaluation_cases = [
    {
        "question": "What is Apache Spark?",
        "reference": "Apache Spark is a distributed data processing engine used for large-scale data processing."
    },
    {
        "question": "What is Kafka?",
        "reference": "Apache Kafka is a distributed event streaming platform used to publish, store, and consume streams of records."
    },
    {
        "question": "What is Airflow?",
        "reference": "Apache Airflow is a platform for programmatically authoring, scheduling, and monitoring workflows."
    }
]


# Process every evaluation case.
for index, case in enumerate(evaluation_cases, start=1):

    # Display the test case number.
    print(f"\n{'=' * 60}")
    print(f"TEST CASE {index}")
    print(f"{'=' * 60}")

    # Display the question.
    print(f"\nQuestion:\n{case['question']}")

    # Generate an answer using Ollama.
    generated_answer = ask_ollama(case["question"])

    # Display the generated answer.
    print(f"\nGenerated Answer:\n{generated_answer}")

    # Display the trusted reference answer.
    print(f"\nReference Answer:\n{case['reference']}")

    # Calculate exact-match score.
    exact_score = exact_match(
        generated_answer,
        case["reference"]
    )

    # Calculate semantic similarity score.
    semantic_score = semantic_similarity(
        generated_answer,
        case["reference"]
    )

    # Display the evaluation scores.
    print(f"\nExact Match Score: {exact_score:.2f}")
    print(f"Semantic Similarity Score: {semantic_score:.2f}")