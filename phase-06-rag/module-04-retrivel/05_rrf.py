# ============================================================
# Practical 5: Reciprocal Rank Fusion (RRF)
# Dense Retrieval + BM25 + Rank Fusion
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
# 2. Configuration
# ============================================================

EMBEDDING_MODEL = "nomic-embed-text"

TOP_K = 3

RRF_K = 60


# ============================================================
# 3. Create Embedding
# ============================================================

def create_embedding(text):
    """Create a dense vector embedding for the given text."""

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


# Normalize vectors so inner product becomes cosine similarity.

faiss.normalize_L2(
    document_embeddings
)


# Get embedding dimension.

dimension = document_embeddings.shape[1]


# Create exact FAISS index using inner product.

dense_index = faiss.IndexFlatIP(
    dimension
)


# Store document embeddings inside FAISS.

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
# 5. Tokenization
# ============================================================

def tokenize(text):
    """Convert text into normalized word tokens for BM25."""

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    return text.split()


# ============================================================
# 6. Build BM25 Index
# ============================================================

print("\n" + "=" * 60)
print("BUILDING BM25 RETRIEVER")
print("=" * 60)


# Tokenize every document for BM25.

tokenized_documents = [
    tokenize(document)
    for document in documents
]


# Create BM25 index from tokenized documents.

bm25 = BM25Okapi(
    tokenized_documents
)


print(
    f"Number of documents: {len(documents)}"
)


# ============================================================
# 7. Dense Search
# ============================================================

def dense_search(query):
    """Retrieve documents ranked by cosine similarity."""

    # Convert query into embedding.

    query_embedding = create_embedding(
        query
    )


    # Convert query embedding to NumPy.

    query_vector = np.array(
        [query_embedding],
        dtype="float32"
    )


    # Normalize query vector.

    faiss.normalize_L2(
        query_vector
    )


    # Search Dense index.

    similarities, indices = dense_index.search(
        query_vector,
        TOP_K
    )


    return (
        similarities[0],
        indices[0]
    )


# ============================================================
# 8. BM25 Search
# ============================================================

def bm25_search(query):
    """Retrieve documents ranked by BM25 lexical relevance."""

    # Tokenize the user query.

    tokenized_query = tokenize(
        query
    )


    # Calculate BM25 score for every document.

    scores = bm25.get_scores(
        tokenized_query
    )


    # Sort document IDs by BM25 score.

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:TOP_K]


    # Get scores corresponding to top documents.

    top_scores = [
        scores[index_id]
        for index_id in top_indices
    ]


    return (
        np.array(top_scores),
        np.array(top_indices)
    )


# ============================================================
# 9. RRF Calculation
# ============================================================

def calculate_rrf(dense_indices, bm25_indices):
    """
    Combine Dense and BM25 rankings using Reciprocal Rank Fusion.
    """

    # Dictionary storing final RRF score for every document.

    rrf_scores = {}


    # --------------------------------------------------------
    # Process Dense ranking
    # --------------------------------------------------------

    for rank, index_id in enumerate(
        dense_indices,
        start=1
    ):

        index_id = int(index_id)

        # RRF contribution = 1 / (k + rank)

        contribution = (
            1
            /
            (RRF_K + rank)
        )


        # Add Dense contribution.

        rrf_scores[index_id] = (
            rrf_scores.get(index_id, 0)
            +
            contribution
        )


    # --------------------------------------------------------
    # Process BM25 ranking
    # --------------------------------------------------------

    for rank, index_id in enumerate(
        bm25_indices,
        start=1
    ):

        index_id = int(index_id)

        # RRF contribution = 1 / (k + rank)

        contribution = (
            1
            /
            (RRF_K + rank)
        )


        # Add BM25 contribution.

        rrf_scores[index_id] = (
            rrf_scores.get(index_id, 0)
            +
            contribution
        )


    # Sort documents by final RRF score.

    ranked_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


    return ranked_results


# ============================================================
# 10. Run Experiment
# ============================================================

query = "What paperwork is needed for a home loan?"


print("\n" + "=" * 60)
print("QUERY")
print("=" * 60)

print(query)


# ============================================================
# 11. Dense Retrieval
# ============================================================

dense_scores, dense_indices = dense_search(
    query
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


# ============================================================
# 12. BM25 Retrieval
# ============================================================

bm25_scores, bm25_indices = bm25_search(
    query
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


# ============================================================
# 13. RRF
# ============================================================

rrf_results = calculate_rrf(
    dense_indices,
    bm25_indices
)


print("\n--- RRF Results ---")


for rank, (index_id, score) in enumerate(
    rrf_results,
    start=1
):

    print(
        f"Rank {rank} | "
        f"ID {index_id} | "
        f"RRF Score: {score:.6f}"
    )

    print(
        f"Document: {documents[index_id]}"
    )


# ============================================================
# 14. Detailed RRF Calculation
# ============================================================

print("\n" + "=" * 60)
print("DETAILED RRF CALCULATION")
print("=" * 60)


# Create rank lookup for Dense.

dense_rank = {
    int(index_id): rank
    for rank, index_id
    in enumerate(
        dense_indices,
        start=1
    )
}


# Create rank lookup for BM25.

bm25_rank = {
    int(index_id): rank
    for rank, index_id
    in enumerate(
        bm25_indices,
        start=1
    )
}


# Every document appearing in either ranking.

all_ids = (
    set(dense_rank.keys())
    |
    set(bm25_rank.keys())
)


for index_id in sorted(all_ids):

    # Get Dense rank.

    d_rank = dense_rank.get(
        index_id
    )


    # Get BM25 rank.

    b_rank = bm25_rank.get(
        index_id
    )


    # Dense contribution.

    dense_contribution = 0

    if d_rank is not None:

        dense_contribution = (
            1
            /
            (RRF_K + d_rank)
        )


    # BM25 contribution.

    bm25_contribution = 0

    if b_rank is not None:

        bm25_contribution = (
            1
            /
            (RRF_K + b_rank)
        )


    # Final RRF score.

    final_score = (
        dense_contribution
        +
        bm25_contribution
    )


    print(
        f"\nID {index_id}"
    )

    print(
        f"Dense Rank: {d_rank}"
    )

    print(
        f"BM25 Rank: {b_rank}"
    )

    print(
        f"Dense Contribution: "
        f"{dense_contribution:.6f}"
    )

    print(
        f"BM25 Contribution: "
        f"{bm25_contribution:.6f}"
    )

    print(
        f"RRF Score: "
        f"{final_score:.6f}"
    )

    print(
        f"Document: {documents[index_id]}"
    )


print("\n" + "=" * 60)
print("RRF COMPLETED")
print("=" * 60)