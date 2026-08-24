from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. Load models
# =========================================================

bi_encoder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# =========================================================
# 2. Knowledge-base chunks
# =========================================================

chunks = [
    "Spark executors can fail when available memory is insufficient.",

    "Large partitions can increase memory pressure on Spark executors.",

    "Kafka consumer lag increases when message processing is slow.",

    "Airflow tasks can fail because of dependency or configuration errors.",

    "Spark jobs can fail when executors run out of memory.",

    "Increasing executor memory can help when Spark workloads require more memory.",

    "Kafka brokers store messages in partitions.",

    "Spark driver memory is different from Spark executor memory."
]


# =========================================================
# 3. User query
# =========================================================

query = "Why did my Spark executor run out of memory?"


# =========================================================
# STAGE 1 — BI-ENCODER RETRIEVAL
# =========================================================

print("\n========== STAGE 1: BI-ENCODER ==========\n")


# Create document embeddings
document_embeddings = bi_encoder.encode(chunks)


# Create query embedding
query_embedding = bi_encoder.encode(query)


# Calculate similarity
scores = cosine_similarity(
    [query_embedding],
    document_embeddings
)[0]


# Pair chunks with similarity scores
bi_results = list(
    zip(chunks, scores)
)


# Sort highest similarity first
bi_results = sorted(
    bi_results,
    key=lambda x: x[1],
    reverse=True
)


# Select initial candidates
candidate_k = 5

candidates = bi_results[:candidate_k]


print("Initial candidates:\n")

for rank, (chunk, score) in enumerate(
    candidates,
    start=1
):
    print(f"Rank {rank}")
    print(f"Bi-Encoder score: {score:.4f}")
    print(f"Chunk: {chunk}")
    print("----------------------------------------")


# =========================================================
# STAGE 2 — CROSS-ENCODER RERANKING
# =========================================================

print("\n========== STAGE 2: CROSS-ENCODER ==========\n")


# Create query-document pairs
pairs = [
    (query, chunk)
    for chunk, _ in candidates
]


# Calculate Cross-Encoder scores
rerank_scores = cross_encoder.predict(pairs)


# Combine candidate chunks with Cross-Encoder scores
reranked_results = list(
    zip(
        [chunk for chunk, _ in candidates],
        rerank_scores
    )
)


# Sort by Cross-Encoder score
reranked_results = sorted(
    reranked_results,
    key=lambda x: x[1],
    reverse=True
)


# =========================================================
# 4. Final results
# =========================================================

print("Reranked results:\n")

for rank, (chunk, score) in enumerate(
    reranked_results,
    start=1
):

    print(f"Rank {rank}")
    print(f"Cross-Encoder score: {score:.4f}")
    print(f"Chunk: {chunk}")
    print("----------------------------------------")


# =========================================================
# 5. Final Top-K for LLM
# =========================================================

final_k = 3

final_results = reranked_results[:final_k]

print("\n========== FINAL CONTEXT ==========\n")

for rank, (chunk, score) in enumerate(
    final_results,
    start=1
):

    print(f"{rank}. {chunk}")