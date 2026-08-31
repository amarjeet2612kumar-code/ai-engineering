from sentence_transformers import SentenceTransformer
import numpy as np


# Load the embedding model
model = SentenceTransformer("all-mpnet-base-v2")


# Knowledge base
documents = [
    {
        "id": 1,
        "text": [
            "Home loan applicants must meet the bank's eligibility criteria.",
            "Applicants should have a stable source of income.",
            "Applicants must submit identity proof, address proof and income documents.",
            "These documents may include salary slips and bank statements."
        ]
    },
    {
        "id": 2,
        "text": [
            "Home loan approval depends on the applicant's credit score.",
            "Existing financial liabilities may also affect approval."
        ]
    },
    {
        "id": 3,
        "text": [
            "Personal loans can be used for various personal expenses.",
            "They are different from home loans."
        ]
    }
]


# Flatten all sentences for retrieval
sentences = []

for document in documents:

    for position, text in enumerate(document["text"]):

        sentences.append(
            {
                "document_id": document["id"],
                "position": position,
                "text": text
            }
        )


# Create sentence embeddings
sentence_embeddings = model.encode(
    [sentence["text"] for sentence in sentences],
    normalize_embeddings=True
)


def choose_strategy(query):
    """Choose the retrieval strategy based on the query."""

    query = query.lower()

    if "depends" in query or "relationship" in query:
        return "graph"

    return "dense"


def dense_retrieval(query, top_k=1):
    """Find the most relevant sentence using semantic similarity."""

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = np.dot(
        sentence_embeddings,
        query_embedding
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "index": index,
            "score": scores[index]
        }
        for index in top_indices
    ]


def get_sentence_window(index, window_size=1):
    """Return the matched sentence and its nearby sentences."""

    sentence = sentences[index]

    document_id = sentence["document_id"]
    position = sentence["position"]

    window = []

    for item in sentences:

        if item["document_id"] != document_id:
            continue

        if abs(item["position"] - position) <= window_size:
            window.append(item)

    return window


def check_relevance(score, threshold=0.40):
    """Check whether the retrieved sentence is relevant enough."""

    return score >= threshold


# User query
query = "What documents are required for a home loan?"


print("=" * 70)
print("USER QUERY")
print("=" * 70)

print(query)


# Step 1: Adaptive routing
strategy = choose_strategy(query)


print("\n" + "=" * 70)
print("ADAPTIVE ROUTER")
print("=" * 70)

print(f"Selected strategy: {strategy}")


# Step 2: Dense retrieval
results = dense_retrieval(query)


print("\n" + "=" * 70)
print("DENSE RETRIEVAL")
print("=" * 70)

for result in results:

    index = result["index"]

    print(f"Sentence ID: {index}")
    print(f"Score: {result['score']:.4f}")
    print(f"Sentence: {sentences[index]['text']}")


# Step 3: Relevance check
best_result = results[0]

if not check_relevance(best_result["score"]):

    print("\nNo sufficiently relevant context found.")

else:

    # Step 4: Sentence window
    window = get_sentence_window(
        best_result["index"],
        window_size=1
    )


    print("\n" + "=" * 70)
    print("SENTENCE WINDOW")
    print("=" * 70)

    for sentence in window:
        print(sentence["text"])


    # Step 5: Build final context
    context = "\n".join(
        sentence["text"]
        for sentence in window
    )


    print("\n" + "=" * 70)
    print("FINAL CONTEXT")
    print("=" * 70)

    print(context)