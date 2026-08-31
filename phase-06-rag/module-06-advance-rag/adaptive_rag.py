from sentence_transformers import SentenceTransformer
import numpy as np


# Load the embedding model
model = SentenceTransformer("all-mpnet-base-v2")


# Documents used for dense retrieval
documents = [
    "Home loan applicants need identity proof, address proof and income documents.",
    "Home loan eligibility requires sufficient income and a good credit history.",
    "Home loan approval depends on income, credit score and existing liabilities.",
    "Credit cards provide a revolving credit facility to customers.",
    "Personal loans can be used for various personal expenses."
]


# Create document embeddings
document_embeddings = model.encode(
    documents,
    normalize_embeddings=True
)


# Simple graph for relationship-based questions
graph = {
    "Home Loan": {
        "requires": [
            "Identity Proof",
            "Address Proof",
            "Income Documents"
        ],
        "depends_on": [
            "Credit Score"
        ]
    }
}


# Decide which retrieval strategy should be used
def choose_strategy(query):
    """Choose a retrieval strategy based on the query."""

    query = query.lower()

    if "relationship" in query or "depends" in query:
        return "graph"

    if "keyword" in query:
        return "bm25"

    return "dense"


# Perform dense retrieval
def dense_retrieval(query, top_k=3):
    """Find documents using semantic similarity."""

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = np.dot(
        document_embeddings,
        query_embedding
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "id": index,
            "score": scores[index],
            "document": documents[index]
        }
        for index in top_indices
    ]


# Perform simple graph retrieval
def graph_retrieval(query):
    """Retrieve connected information from the knowledge graph."""

    if "home loan" not in query.lower():
        return []

    relationships = graph["Home Loan"]

    results = []

    for relation, entities in relationships.items():

        for entity in entities:

            results.append(
                f"Home Loan {relation} {entity}"
            )

    return results


# User query
query = "What documents are required for a home loan?"


print("=" * 70)
print("USER QUERY")
print("=" * 70)

print(query)


# Step 1: Choose the retrieval strategy
strategy = choose_strategy(query)


print("\n" + "=" * 70)
print("ADAPTIVE DECISION")
print("=" * 70)

print(f"Selected strategy: {strategy}")


# Step 2: Execute the selected strategy
print("\n" + "=" * 70)
print("RETRIEVAL RESULTS")
print("=" * 70)


if strategy == "dense":

    results = dense_retrieval(query)

    for rank, result in enumerate(results, 1):

        print(f"\nRank: {rank}")
        print(f"ID: {result['id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Document: {result['document']}")


elif strategy == "graph":

    results = graph_retrieval(query)

    for result in results:
        print(result)


elif strategy == "bm25":

    print("BM25 retrieval would be executed here.")