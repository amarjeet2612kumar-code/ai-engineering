# ============================================================
# Practical 3: Dense Retrieval vs BM25
# ============================================================

import re

import faiss
import numpy as np

import ollama
from rank_bm25 import BM25Okapi


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
# 2. Embedding Model
# ============================================================

EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# 3. Dense Retrieval
# ============================================================

def create_embedding(text):

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response["embeddings"][0]


print("=" * 60)
print("BUILDING DENSE RETRIEVER")
print("=" * 60)


# Create document embeddings

document_embeddings = []

for document in documents:

    embedding = create_embedding(document)

    document_embeddings.append(embedding)


document_embeddings = np.array(
    document_embeddings,
    dtype="float32"
)


# Normalize document vectors

faiss.normalize_L2(
    document_embeddings
)


# Create FAISS index

dimension = document_embeddings.shape[1]

dense_index = faiss.IndexFlatIP(
    dimension
)


# Add vectors

dense_index.add(
    document_embeddings
)


print(
    f"Number of vectors: {dense_index.ntotal}"
)

print(
    f"Vector dimension: {dimension}"
)


# ============================================================
# 4. Sparse Retrieval - BM25
# ============================================================

print("\n" + "=" * 60)
print("BUILDING BM25 RETRIEVER")
print("=" * 60)


def tokenize(text):

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    return text.split()


tokenized_documents = [
    tokenize(document)
    for document in documents
]


bm25 = BM25Okapi(
    tokenized_documents
)


print(
    f"Number of documents: {len(documents)}"
)


# ============================================================
# 5. Dense Search Function
# ============================================================

def dense_search(query, k=3):

    # Create query embedding

    query_embedding = create_embedding(
        query
    )


    # Convert to NumPy

    query_vector = np.array(
        [query_embedding],
        dtype="float32"
    )


    # Normalize query

    faiss.normalize_L2(
        query_vector
    )


    # Search

    similarities, indices = dense_index.search(
        query_vector,
        k
    )


    return similarities[0], indices[0]


# ============================================================
# 6. BM25 Search Function
# ============================================================

def bm25_search(query, k=3):

    # Tokenize query

    tokenized_query = tokenize(
        query
    )


    # Calculate BM25 scores

    scores = bm25.get_scores(
        tokenized_query
    )


    # Sort by score descending

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]


    return (
        [scores[i] for i in top_indices],
        top_indices
    )


# ============================================================
# 7. Compare Both Retrievers
# ============================================================

def compare_retrievers(query, k=3):

    print("\n")
    print("=" * 60)
    print("QUERY")
    print("=" * 60)

    print(query)


    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    dense_scores, dense_indices = dense_search(
        query,
        k
    )


    print("\n--- Dense Retrieval ---")

    for rank, (score, index_id) in enumerate(
        zip(dense_scores, dense_indices),
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


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_scores, bm25_indices = bm25_search(
        query,
        k
    )


    print("\n--- BM25 Retrieval ---")

    for rank, (score, index_id) in enumerate(
        zip(bm25_scores, bm25_indices),
        start=1
    ):

        print(
            f"Rank {rank} | "
            f"ID {index_id} | "
            f"BM25: {score:.4f}"
        )

        print(
            f"Document: {documents[index_id]}"
        )


    # --------------------------------------------------------
    # Compare IDs
    # --------------------------------------------------------

    dense_set = set(
        dense_indices
    )

    bm25_set = set(
        bm25_indices
    )


    common = dense_set.intersection(
        bm25_set
    )


    dense_only = dense_set.difference(
        bm25_set
    )


    bm25_only = bm25_set.difference(
        dense_set
    )


    print("\n--- Comparison ---")

    print(
        "Common IDs:",
        common
    )

    print(
        "Dense-only IDs:",
        dense_only
    )

    print(
        "BM25-only IDs:",
        bm25_only
    )


# ============================================================
# 8. Test Queries
# ============================================================

queries = [

    # Semantic wording
    "What paperwork is needed for a home loan?",

    # Exact terminology
    "credit card",

    # Semantic + terminology variation
    "housing finance documentation"
]


# ============================================================
# 9. Run Experiments
# ============================================================

for query in queries:

    compare_retrievers(
        query,
        k=3
    )


print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED")
print("=" * 60)