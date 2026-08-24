from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. Load a Matryoshka-compatible embedding model
# =========================================================

model = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1"
)


# =========================================================
# 2. Documents
# =========================================================

documents = [
    "Spark executors can fail when available memory is insufficient.",

    "Large Spark partitions can increase executor memory pressure.",

    "Kafka consumer lag increases when message processing is slow.",

    "Airflow tasks can fail because of dependency configuration errors.",

    "Spark jobs can fail when executors run out of memory."
]


# =========================================================
# 3. Query
# =========================================================

query = "Why did my Spark executor run out of memory?"


# =========================================================
# 4. Generate the FULL embedding
# =========================================================

full_embeddings = model.encode(
    documents
)

query_embedding = model.encode(
    query
)


print("\n========== EMBEDDING INFORMATION ==========\n")

print("Full document dimension:")
print(full_embeddings.shape)

print("\nFull query dimension:")
print(query_embedding.shape)


# =========================================================
# 5. Test different dimensions
# =========================================================

dimensions = [1024, 512, 256, 128]


for dimension in dimensions:

    print("\n========================================")
    print(f"Dimension: {dimension}")
    print("========================================")

    # ---------------------------------------------
    # Truncate embeddings
    # ---------------------------------------------

    document_vectors = full_embeddings[
        :, :dimension
    ]

    query_vector = query_embedding[
        :dimension
    ]


    # ---------------------------------------------
    # Calculate similarity
    # ---------------------------------------------

    scores = cosine_similarity(
        [query_vector],
        document_vectors
    )[0]


    # ---------------------------------------------
    # Rank documents
    # ---------------------------------------------

    results = list(
        zip(documents, scores)
    )

    results = sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )


    # ---------------------------------------------
    # Display ranking
    # ---------------------------------------------

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{score:.4f} | "
            f"{document}"
        )