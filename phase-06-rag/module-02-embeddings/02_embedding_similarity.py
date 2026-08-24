from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Load embedding model
# ---------------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# 2. Sentences
# ---------------------------------------------------------

sentences = [
    "Spark executor ran out of memory",
    "The Spark executor failed because of insufficient memory",
    "Large executor memory pressure caused the job to fail",
    "Kafka consumer is experiencing high lag",
    "Cricket is a popular sport in India"
]


# ---------------------------------------------------------
# 3. Generate embeddings
# ---------------------------------------------------------

embeddings = model.encode(sentences)


# ---------------------------------------------------------
# 4. Compare every sentence with every other sentence
# ---------------------------------------------------------

similarity_matrix = cosine_similarity(
    embeddings
)


# ---------------------------------------------------------
# 5. Display results
# ---------------------------------------------------------

print("\n========== SENTENCES ==========\n")

for i, sentence in enumerate(sentences):
    print(f"{i}: {sentence}")


print("\n========== SIMILARITY MATRIX ==========\n")

print(similarity_matrix)


# ---------------------------------------------------------
# 4. Compare sentence pairs
# ---------------------------------------------------------

pairs = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
]


print("\n========== PAIR SIMILARITY ==========\n")

for i, j in pairs:

    score = cosine_similarity(
        [embeddings[i]],
        [embeddings[j]]
    )[0][0]

    print(f"Sentence {i}")
    print(sentences[i])

    print(f"\nSentence {j}")
    print(sentences[j])

    print(f"\nSimilarity: {score:.4f}")
    print("----------------------------------------")