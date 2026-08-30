# ============================================================
# Practical 7: Multi-Query Retrieval
# ============================================================
#
# Goal:
# Generate multiple search queries from one user question,
# retrieve documents for each query, and combine the results.
# ============================================================

import faiss
import numpy as np
import ollama


# ============================================================
# 1. Documents
# ============================================================

documents = [
    "Home loan eligibility requires sufficient income and a good credit history.",

    "Home loan applicants need identity proof, address proof and income documents.",

    "Personal loans can be used for various personal expenses.",

    "Home loan interest rates depend on the applicant's profile and loan tenure.",

    "Credit cards provide a revolving credit facility to customers.",

    "Home loan applicants should submit salary slips and bank statements.",

    "Home loan approval depends on income, credit score and existing liabilities."
]


# ============================================================
# 2. Configuration
# ============================================================

EMBEDDING_MODEL = "nomic-embed-text"

LLM_MODEL = "llama3.2:3b"

TOP_K = 3

NUMBER_OF_QUERIES = 3


# ============================================================
# 3. Create Embedding
# ============================================================

def create_embedding(text):
    """Convert text into a dense vector for semantic retrieval."""

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response["embeddings"][0]


# ============================================================
# 4. Build Dense Index
# ============================================================

print("=" * 60)
print("BUILDING DENSE RETRIEVER")
print("=" * 60)


# Create embeddings for all documents.

document_embeddings = []

for document in documents:

    embedding = create_embedding(
        document
    )

    document_embeddings.append(
        embedding
    )


# Convert embeddings into NumPy array.

document_embeddings = np.array(
    document_embeddings,
    dtype="float32"
)


# Normalize vectors for cosine similarity.

faiss.normalize_L2(
    document_embeddings
)


# Determine vector dimension.

dimension = document_embeddings.shape[1]


# Create FAISS index.

index = faiss.IndexFlatIP(
    dimension
)


# Add document vectors.

index.add(
    document_embeddings
)


print(
    f"Number of vectors: {index.ntotal}"
)

print(
    f"Vector dimension: {dimension}"
)


# ============================================================
# 5. Generate Multiple Queries
# ============================================================

def generate_queries(user_query):
    """Generate multiple search queries representing different aspects of the question."""

    prompt = f"""
Generate exactly {NUMBER_OF_QUERIES} different search queries
for retrieving information relevant to the user's question.

Rules:
- Each query must preserve the user's intent.
- Each query should focus on a different aspect or wording.
- Queries must be useful for document retrieval.
- Do not answer the question.
- Return one query per line.
- Do not number the queries.

User question:
{user_query}

Search queries:
"""


    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    raw_output = response["message"]["content"].strip()


    # Convert LLM output into individual queries.

    queries = [
        line.strip()
        for line in raw_output.split("\n")
        if line.strip()
    ]


    # Remove accidental numbering such as "1." or "-".

    cleaned_queries = []

    for query in queries:

        query = query.lstrip(
            "0123456789.-) "
        )

        if query:
            cleaned_queries.append(
                query
            )


    # Keep only requested number of queries.

    return cleaned_queries[:NUMBER_OF_QUERIES]


# ============================================================
# 6. Dense Search
# ============================================================

def dense_search(query):
    """Retrieve the top documents that are semantically similar to one query."""

    # Convert query into embedding.

    query_embedding = create_embedding(
        query
    )


    # Convert into NumPy array.

    query_vector = np.array(
        [query_embedding],
        dtype="float32"
    )


    # Normalize query vector.

    faiss.normalize_L2(
        query_vector
    )


    # Search vector index.

    similarities, indices = index.search(
        query_vector,
        TOP_K
    )


    return similarities[0], indices[0]


# ============================================================
# 7. User Query
# ============================================================

user_query = (
    "What should I know before applying for a home loan?"
)


print("\n" + "=" * 60)
print("USER QUERY")
print("=" * 60)

print(user_query)


# ============================================================
# 8. Generate Multiple Queries
# ============================================================

generated_queries = generate_queries(
    user_query
)


print("\n" + "=" * 60)
print("GENERATED QUERIES")
print("=" * 60)


for number, query in enumerate(
    generated_queries,
    start=1
):

    print(
        f"Query {number}: {query}"
    )


# ============================================================
# 9. Retrieve For Every Query
# ============================================================

print("\n" + "=" * 60)
print("MULTI-QUERY RETRIEVAL")
print("=" * 60)


all_results = []


for query_number, query in enumerate(
    generated_queries,
    start=1
):

    scores, indices = dense_search(
        query
    )


    print(
        f"\n--- Query {query_number} ---"
    )

    print(query)


    for rank, (score, index_id) in enumerate(
        zip(scores, indices),
        start=1
    ):

        index_id = int(index_id)


        print(
            f"Rank {rank} | "
            f"ID {index_id} | "
            f"Cosine: {score:.4f}"
        )

        print(
            f"Document: {documents[index_id]}"
        )


        # Store result for later combination.

        all_results.append(
            {
                "query_number": query_number,
                "rank": rank,
                "id": index_id,
                "score": float(score)
            }
        )


# ============================================================
# 10. Combine Results
# ============================================================

def combine_results(results):
    """Combine results from multiple queries by counting how often each document appears."""

    document_stats = {}


    for result in results:

        document_id = result["id"]


        if document_id not in document_stats:

            document_stats[document_id] = {
                "count": 0,
                "best_score": 0.0
            }


        # Count how many generated queries retrieved the document.

        document_stats[document_id]["count"] += 1


        # Keep the best similarity score seen for this document.

        document_stats[document_id]["best_score"] = max(
            document_stats[document_id]["best_score"],
            result["score"]
        )


    # Sort primarily by frequency and secondarily by similarity.

    ranked_results = sorted(
        document_stats.items(),
        key=lambda item: (
            item[1]["count"],
            item[1]["best_score"]
        ),
        reverse=True
    )


    return ranked_results


# ============================================================
# 11. Final Combined Results
# ============================================================

combined_results = combine_results(
    all_results
)


print("\n" + "=" * 60)
print("COMBINED RESULTS")
print("=" * 60)


for rank, (document_id, stats) in enumerate(
    combined_results,
    start=1
):

    print(
        f"\nFinal Rank: {rank}"
    )

    print(
        f"ID: {document_id}"
    )

    print(
        f"Retrieved by "
        f"{stats['count']} query(s)"
    )

    print(
        f"Best Cosine Similarity: "
        f"{stats['best_score']:.4f}"
    )

    print(
        f"Document: {documents[document_id]}"
    )


print("\n" + "=" * 60)
print("MULTI-QUERY RETRIEVAL COMPLETED")
print("=" * 60)