# ============================================================
# Practical 8: HyDE (Hypothetical Document Embeddings)
# ============================================================
#
# Goal:
# Compare normal dense retrieval with HyDE retrieval.
#
# Normal:
# User Query → Embedding → Vector Search
#
# HyDE:
# User Query → LLM → Hypothetical Document
#            → Embedding → Vector Search
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
# 4. Build Vector Index
# ============================================================

print("=" * 60)
print("BUILDING DENSE RETRIEVER")
print("=" * 60)


document_embeddings = []


for document in documents:

    embedding = create_embedding(
        document
    )

    document_embeddings.append(
        embedding
    )


document_embeddings = np.array(
    document_embeddings,
    dtype="float32"
)


# Normalize document vectors so inner product = cosine similarity.

faiss.normalize_L2(
    document_embeddings
)


dimension = document_embeddings.shape[1]


index = faiss.IndexFlatIP(
    dimension
)


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
# 5. Dense Retrieval
# ============================================================

def dense_search(query):
    """Search the vector index using the embedding of the supplied text."""

    query_embedding = create_embedding(
        query
    )


    query_vector = np.array(
        [query_embedding],
        dtype="float32"
    )


    faiss.normalize_L2(
        query_vector
    )


    similarities, indices = index.search(
        query_vector,
        TOP_K
    )


    return similarities[0], indices[0]


# ============================================================
# 6. Generate Hypothetical Document
# ============================================================

def generate_hypothetical_document(query):
    """Ask the LLM to generate a hypothetical document that could answer the query."""

    prompt = f"""
Write a short hypothetical document that could contain
the answer to the following user question.

The document is only for information retrieval.
Do not mention that it is hypothetical.
Include relevant concepts and terminology.

Do not add unnecessary explanation.

User question:
{query}

Hypothetical document:
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


    hypothetical_document = (
        response["message"]["content"]
        .strip()
    )


    return hypothetical_document


# ============================================================
# 7. User Query
# ============================================================

query = (
    "What documents are required for a home loan?"
)


print("\n" + "=" * 60)
print("USER QUERY")
print("=" * 60)

print(query)


# ============================================================
# 8. Normal Dense Retrieval
# ============================================================

normal_scores, normal_indices = dense_search(
    query
)


print("\n" + "=" * 60)
print("NORMAL DENSE RETRIEVAL")
print("=" * 60)


for rank, (score, index_id) in enumerate(
    zip(normal_scores, normal_indices),
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


# ============================================================
# 9. Generate Hypothetical Document
# ============================================================

hypothetical_document = (
    generate_hypothetical_document(
        query
    )
)


print("\n" + "=" * 60)
print("HYDE")
print("=" * 60)


print("\nHypothetical Document:")

print(
    hypothetical_document
)


# ============================================================
# 10. HyDE Retrieval
# ============================================================

hyde_scores, hyde_indices = dense_search(
    hypothetical_document
)


print("\n--- HyDE Retrieval ---")


for rank, (score, index_id) in enumerate(
    zip(hyde_scores, hyde_indices),
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


# ============================================================
# 11. Compare Results
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)


normal_ids = set(
    int(index_id)
    for index_id in normal_indices
)


hyde_ids = set(
    int(index_id)
    for index_id in hyde_indices
)


print(
    f"Normal Dense IDs: {normal_ids}"
)


print(
    f"HyDE IDs: {hyde_ids}"
)


print(
    f"Common IDs: "
    f"{normal_ids.intersection(hyde_ids)}"
)


print(
    f"Normal-only IDs: "
    f"{normal_ids - hyde_ids}"
)


print(
    f"HyDE-only IDs: "
    f"{hyde_ids - normal_ids}"
)


print("\n" + "=" * 60)
print("HYDE RETRIEVAL COMPLETED")
print("=" * 60)