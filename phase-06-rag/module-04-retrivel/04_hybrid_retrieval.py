# ============================================================
# Practical 4: Hybrid Retrieval
# Dense Retrieval + BM25
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

K = 3

DENSE_WEIGHT = 0.5

BM25_WEIGHT = 0.5


# ============================================================
# 3. Embedding Function
# ============================================================

def create_embedding(text):

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response["embeddings"][0]


# ============================================================
# 4. Build Dense Retriever
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


# Normalize vectors for cosine similarity

faiss.normalize_L2(
    document_embeddings
)


dimension = document_embeddings.shape[1]


dense_index = faiss.IndexFlatIP(
    dimension
)


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
# 5. Build BM25 Retriever
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
# 6. Min-Max Normalization
# ============================================================

def min_max_normalize(scores):

    scores = np.array(
        scores,
        dtype="float32"
    )

    minimum = scores.min()

    maximum = scores.max()


    # Avoid division by zero

    if maximum == minimum:

        return np.ones_like(
            scores
        )


    return (
        (scores - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# 7. Dense Search
# ============================================================

def dense_search(query):

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


    similarities, indices = dense_index.search(
        query_vector,
        K
    )


    return (
        similarities[0],
        indices[0]
    )


# ============================================================
# 8. BM25 Search
# ============================================================

def bm25_search(query):

    tokenized_query = tokenize(
        query
    )


    scores = bm25.get_scores(
        tokenized_query
    )


    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:K]


    top_scores = [
        scores[i]
        for i in top_indices
    ]


    return (
        np.array(top_scores),
        np.array(top_indices)
    )


# ============================================================
# 9. Hybrid Search
# ============================================================

def hybrid_search(query):

    # --------------------------------------------------------
    # Dense Retrieval
    # --------------------------------------------------------

    dense_scores, dense_indices = dense_search(
        query
    )


    # --------------------------------------------------------
    # BM25 Retrieval
    # --------------------------------------------------------

    bm25_scores, bm25_indices = bm25_search(
        query
    )


    # --------------------------------------------------------
    # Create score dictionaries
    # --------------------------------------------------------

    dense_score_map = {
        int(index_id): float(score)
        for index_id, score
        in zip(dense_indices, dense_scores)
    }


    bm25_score_map = {
        int(index_id): float(score)
        for index_id, score
        in zip(bm25_indices, bm25_scores)
    }


    # --------------------------------------------------------
    # Collect all candidate document IDs
    # --------------------------------------------------------

    candidate_ids = (
        set(dense_indices)
        |
        set(bm25_indices)
    )


    candidate_ids = [
        int(index_id)
        for index_id in candidate_ids
    ]


    # --------------------------------------------------------
    # Get scores for candidates
    # --------------------------------------------------------

    dense_candidate_scores = [
        dense_score_map.get(
            index_id,
            0.0
        )
        for index_id in candidate_ids
    ]


    bm25_candidate_scores = [
        bm25_score_map.get(
            index_id,
            0.0
        )
        for index_id in candidate_ids
    ]


    # --------------------------------------------------------
    # Normalize scores
    # --------------------------------------------------------

    normalized_dense = min_max_normalize(
        dense_candidate_scores
    )


    normalized_bm25 = min_max_normalize(
        bm25_candidate_scores
    )


    # --------------------------------------------------------
    # Weighted combination
    # --------------------------------------------------------

    hybrid_scores = (
        DENSE_WEIGHT * normalized_dense
        +
        BM25_WEIGHT * normalized_bm25
    )


    # --------------------------------------------------------
    # Sort final results
    # --------------------------------------------------------

    ranked_positions = np.argsort(
        hybrid_scores
    )[::-1]


    # --------------------------------------------------------
    # Display Dense Results
    # --------------------------------------------------------

    print("\n--- Dense Retrieval ---")

    for rank, index_id in enumerate(
        dense_indices,
        start=1
    ):

        print(
            f"Rank {rank} | "
            f"ID {index_id} | "
            f"Cosine: {dense_score_map[int(index_id)]:.4f}"
        )

        print(
            f"Document: {documents[index_id]}"
        )


    # --------------------------------------------------------
    # Display BM25 Results
    # --------------------------------------------------------

    print("\n--- BM25 Retrieval ---")

    for rank, index_id in enumerate(
        bm25_indices,
        start=1
    ):

        print(
            f"Rank {rank} | "
            f"ID {index_id} | "
            f"BM25: {bm25_score_map[int(index_id)]:.4f}"
        )

        print(
            f"Document: {documents[index_id]}"
        )


    # --------------------------------------------------------
    # Display Hybrid Results
    # --------------------------------------------------------

    print("\n--- Hybrid Retrieval ---")

    for rank, position in enumerate(
        ranked_positions,
        start=1
    ):

        index_id = candidate_ids[position]

        print(
            f"Rank {rank} | "
            f"ID {index_id}"
        )

        print(
            f"Dense Normalized: "
            f"{normalized_dense[position]:.4f}"
        )

        print(
            f"BM25 Normalized: "
            f"{normalized_bm25[position]:.4f}"
        )

        print(
            f"Hybrid Score: "
            f"{hybrid_scores[position]:.4f}"
        )

        print(
            f"Document: {documents[index_id]}"
        )


# ============================================================
# 10. Test Query
# ============================================================

query = "What paperwork is needed for a home loan?"


print("\n" + "=" * 60)
print("QUERY")
print("=" * 60)

print(query)


# ============================================================
# 11. Run Hybrid Retrieval
# ============================================================

hybrid_search(
    query
)


print("\n" + "=" * 60)
print("HYBRID RETRIEVAL COMPLETED")
print("=" * 60)