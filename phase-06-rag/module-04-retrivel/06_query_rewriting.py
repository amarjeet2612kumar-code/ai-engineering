# ============================================================
# Practical 6: Query Rewriting
# ============================================================
#
# Goal:
# Convert a user's natural-language question into a
# retrieval-friendly search query before performing retrieval.
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

    "Credit cards provide a revolving credit facility to customers."
]


# ============================================================
# 2. Configuration
# ============================================================

EMBEDDING_MODEL = "nomic-embed-text"

LLM_MODEL = "llama3.2:3b"

TOP_K = 3


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


# Convert embeddings to NumPy array.

document_embeddings = np.array(
    document_embeddings,
    dtype="float32"
)


# Normalize vectors so inner product represents cosine similarity.

faiss.normalize_L2(
    document_embeddings
)


# Get vector dimension.

dimension = document_embeddings.shape[1]


# Create FAISS index.

index = faiss.IndexFlatIP(
    dimension
)


# Add document vectors to the index.

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
# 5. Query Rewriting
# ============================================================

def rewrite_query(query):
    """Use an LLM to transform the user's question into a search query."""

    prompt = f"""
Rewrite the following user question into a concise
search query for retrieving relevant documents.

Rules:
- Preserve the original intent.
- Use important keywords and concepts.
- Do not answer the question.
- Return only the rewritten search query.

User question:
{query}

Rewritten search query:
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


    rewritten_query = response["message"]["content"].strip()


    return rewritten_query


# ============================================================
# 6. Dense Retrieval
# ============================================================

def dense_search(query):
    """Retrieve the most semantically similar documents for a query."""

    # Convert query into embedding.

    query_embedding = create_embedding(
        query
    )


    # Convert to NumPy array.

    query_vector = np.array(
        [query_embedding],
        dtype="float32"
    )


    # Normalize query vector.

    faiss.normalize_L2(
        query_vector
    )


    # Search the vector index.

    similarities, indices = index.search(
        query_vector,
        TOP_K
    )


    return similarities[0], indices[0]


# ============================================================
# 7. Original User Query
# ============================================================

query = "What paperwork do I need for a home loan?"


print("\n" + "=" * 60)
print("ORIGINAL QUERY")
print("=" * 60)

print(query)


# ============================================================
# 8. Retrieve Using Original Query
# ============================================================

original_scores, original_indices = dense_search(
    query
)


print("\n--- Retrieval Using Original Query ---")


for rank, (score, index_id) in enumerate(
    zip(original_scores, original_indices),
    start=1
):

    print(
        f"Rank {rank} | "
        f"ID {index_id} | "
        f"Cosine: {score:.4f}"
    )

    print(
        f"Document: {documents[index_id]}"
    )


# ============================================================
# 9. Rewrite Query
# ============================================================

rewritten_query = rewrite_query(
    query
)


print("\n" + "=" * 60)
print("QUERY REWRITING")
print("=" * 60)

print(
    f"Original Query:\n{query}"
)

print(
    f"\nRewritten Query:\n{rewritten_query}"
)


# ============================================================
# 10. Retrieve Using Rewritten Query
# ============================================================

rewritten_scores, rewritten_indices = dense_search(
    rewritten_query
)


print("\n--- Retrieval Using Rewritten Query ---")


for rank, (score, index_id) in enumerate(
    zip(rewritten_scores, rewritten_indices),
    start=1
):

    print(
        f"Rank {rank} | "
        f"ID {index_id} | "
        f"Cosine: {score:.4f}"
    )

    print(
        f"Document: {documents[index_id]}"
    )


# ============================================================
# 11. Compare Results
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)


original_ids = set(
    int(index_id)
    for index_id in original_indices
)


rewritten_ids = set(
    int(index_id)
    for index_id in rewritten_indices
)


print(
    f"Original Query IDs: {original_ids}"
)

print(
    f"Rewritten Query IDs: {rewritten_ids}"
)


print(
    f"Common IDs: "
    f"{original_ids.intersection(rewritten_ids)}"
)


print(
    f"Original-only IDs: "
    f"{original_ids - rewritten_ids}"
)


print(
    f"Rewritten-only IDs: "
    f"{rewritten_ids - original_ids}"
)


print("\n" + "=" * 60)
print("QUERY REWRITING COMPLETED")
print("=" * 60)