from sentence_transformers import CrossEncoder


# ---------------------------------------------------------
# 1. Load Cross-Encoder model
# ---------------------------------------------------------

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ---------------------------------------------------------
# 2. User query
# ---------------------------------------------------------

query = "Why did my Spark executor run out of memory?"


# ---------------------------------------------------------
# 3. Candidate documents
# ---------------------------------------------------------

documents = [
    "Spark executors can fail when available memory is insufficient.",

    "Large partitions can increase memory pressure on Spark executors.",

    "Kafka consumer lag increases when message processing is slow.",

    "Airflow tasks can fail because of dependency or configuration errors.",

    "Spark jobs can fail when executors run out of memory."
]


# ---------------------------------------------------------
# 4. Create query-document pairs
# ---------------------------------------------------------

pairs = [
    (query, document)
    for document in documents
]


# ---------------------------------------------------------
# 5. Calculate Cross-Encoder scores
# ---------------------------------------------------------

scores = model.predict(pairs)


# ---------------------------------------------------------
# 6. Combine documents and scores
# ---------------------------------------------------------

results = list(
    zip(documents, scores)
)


# ---------------------------------------------------------
# 7. Sort by relevance
# ---------------------------------------------------------

results = sorted(
    results,
    key=lambda x: x[1],
    reverse=True
)


# ---------------------------------------------------------
# 8. Display results
# ---------------------------------------------------------

print("\n========== QUERY ==========\n")
print(query)

print("\n========== CROSS-ENCODER RESULTS ==========\n")

for rank, (document, score) in enumerate(
    results,
    start=1
):

    print(f"Rank {rank}")
    print(f"Score: {score:.4f}")
    print(f"Document: {document}")
    print("----------------------------------------")