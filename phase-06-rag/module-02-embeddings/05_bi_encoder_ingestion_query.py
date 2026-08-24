from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Load embedding model
# ---------------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# 2. Documents / chunks
# ---------------------------------------------------------

chunks = [
    "Spark executors can fail when available memory is insufficient.",
    "Large partitions can increase memory pressure on executors.",
    "Kafka consumer lag increases when message processing is slow.",
    "Airflow tasks can fail because of dependency or configuration errors.",
    "Spark jobs can fail when executors run out of memory."
]


# =========================================================
# INGESTION TIME
# =========================================================

print("\n========== INGESTION ==========\n")

# Generate document embeddings ONCE
document_embeddings = model.encode(chunks)

print("Documents embedded:", len(document_embeddings))
print("Embedding dimension:", len(document_embeddings[0]))


# =========================================================
# QUERY TIME
# =========================================================

query = "Why did my Spark executor run out of memory?"

print("\n========== QUERY ==========\n")
print("Query:", query)


# Generate embedding for the query
query_embedding = model.encode(query)


# ---------------------------------------------------------
# Similarity search
# ---------------------------------------------------------

scores = cosine_similarity(
    [query_embedding],
    document_embeddings
)[0]


# ---------------------------------------------------------
# Pair chunks with scores
# ---------------------------------------------------------

results = list(
    zip(chunks, scores)
)


# ---------------------------------------------------------
# Sort by similarity
# ---------------------------------------------------------

results = sorted(
    results,
    key=lambda x: x[1],
    reverse=True
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n========== RESULTS ==========\n")

for rank, (chunk, score) in enumerate(
    results,
    start=1
):

    print(f"Rank {rank}")
    print(f"Score: {score:.4f}")
    print(f"Chunk: {chunk}")
    print("----------------------------------------")