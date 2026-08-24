from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Load embedding model
# ---------------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# 2. Existing document chunks
# ---------------------------------------------------------

chunks = [
    "Spark executors can fail when available memory is insufficient.",
    "Large partitions can increase memory pressure on executors.",
    "Kafka consumer lag increases when message processing is slow.",
    "Airflow tasks can fail because of dependency or configuration errors.",
    "Spark jobs can fail when executors run out of memory."
]


# ---------------------------------------------------------
# 3. User query
# ---------------------------------------------------------

query = "Why did my Spark executor run out of memory?"


# ---------------------------------------------------------
# 4. Create document embeddings
# ---------------------------------------------------------

document_embeddings = model.encode(chunks)


# ---------------------------------------------------------
# 5. Create query embedding
# ---------------------------------------------------------

query_embedding = model.encode(query)


# ---------------------------------------------------------
# 6. Calculate similarity
# ---------------------------------------------------------

scores = cosine_similarity(
    [query_embedding],
    document_embeddings
)[0]


# ---------------------------------------------------------
# 7. Sort documents by similarity
# ---------------------------------------------------------

ranked_results = sorted(
    zip(chunks, scores),
    key=lambda x: x[1],
    reverse=True
)


# ---------------------------------------------------------
# 8. Display results
# ---------------------------------------------------------

print("\n========== QUERY ==========\n")
print(query)

print("\n========== RETRIEVAL RESULTS ==========\n")

for rank, (chunk, score) in enumerate(
    ranked_results,
    start=1
):
    print(f"Rank {rank}")
    print(f"Score : {score:.4f}")
    print(f"Chunk : {chunk}")
    print("----------------------------------------")