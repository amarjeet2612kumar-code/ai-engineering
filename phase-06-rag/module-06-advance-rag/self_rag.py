from sentence_transformers import SentenceTransformer
import numpy as np


# Load the embedding model
model = SentenceTransformer("all-mpnet-base-v2")


# Knowledge base
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


# Decide whether the query needs retrieval
def needs_retrieval(query):
    """Decide whether the query needs information from the knowledge base."""

    simple_questions = [
        "what is 2 + 2",
        "what is 1 + 1",
        "what is the capital of india"
    ]

    return query.lower().strip() not in simple_questions


# Retrieve relevant documents
def retrieve_documents(query, top_k=3):
    """Find documents most similar to the user query."""

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = np.dot(document_embeddings, query_embedding)

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "id": index,
            "score": scores[index],
            "document": documents[index]
        }
        for index in top_indices
    ]


# Check whether retrieved documents are relevant
def check_relevance(results, threshold=0.50):
    """Keep only documents whose similarity score passes the threshold."""

    return [
        result
        for result in results
        if result["score"] >= threshold
    ]


# User query
query = "What documents are required for a home loan?"


print("=" * 70)
print("USER QUERY")
print("=" * 70)

print(query)


# Step 1: Decide whether retrieval is required
if needs_retrieval(query):

    print("\n" + "=" * 70)
    print("RETRIEVAL DECISION")
    print("=" * 70)

    print("Retrieval is required.")


    # Step 2: Retrieve documents
    results = retrieve_documents(query)


    print("\n" + "=" * 70)
    print("RETRIEVED DOCUMENTS")
    print("=" * 70)

    for result in results:
        print(
            f"\nID: {result['id']}"
            f"\nScore: {result['score']:.4f}"
            f"\nDocument: {result['document']}"
        )


    # Step 3: Check relevance
    relevant_documents = check_relevance(results)


    print("\n" + "=" * 70)
    print("RELEVANT DOCUMENTS")
    print("=" * 70)

    for result in relevant_documents:
        print(
            f"\nID: {result['id']}"
            f"\nScore: {result['score']:.4f}"
            f"\nDocument: {result['document']}"
        )


else:

    print("\n" + "=" * 70)
    print("RETRIEVAL DECISION")
    print("=" * 70)

    print("Retrieval is not required.")

    print("\nAnswer directly using the LLM.")