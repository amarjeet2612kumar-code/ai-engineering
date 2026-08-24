from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# 1. Load pretrained embedding model
# ---------------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# 2. RAG chunks
# ---------------------------------------------------------

chunks = [
    "Spark executors can fail when available memory is insufficient.",
    "Large partitions can increase memory pressure on executors.",
    "Kafka consumer lag increases when message processing is slow.",
    "Airflow tasks can fail because of dependency or configuration errors."
]


# ---------------------------------------------------------
# 3. Generate embeddings for all chunks
# ---------------------------------------------------------

embeddings = model.encode(
    chunks
)


# ---------------------------------------------------------
# 4. Inspect results
# ---------------------------------------------------------

print("\n========== EMBEDDING INFORMATION ==========\n")

print("Number of chunks :", len(chunks))
print("Number of vectors:", len(embeddings))
print("Vector dimension :", len(embeddings[0]))


# ---------------------------------------------------------
# 5. Inspect each chunk
# ---------------------------------------------------------

print("\n========== CHUNKS + EMBEDDINGS ==========\n")

for i, (chunk, embedding) in enumerate(
    zip(chunks, embeddings),
    start=1
):

    print(f"Chunk {i}:")
    print(chunk)

    print("\nFirst 10 dimensions:")
    print(embedding[:10])

    print("\n----------------------------------------")