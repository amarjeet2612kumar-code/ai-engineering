from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Load document
# ---------------------------------------------------------

document_path = Path(
    "phase-06-rag/module-01-document-processing/"
    "documents/spark_troubleshooting.txt"
)

text = document_path.read_text(
    encoding="utf-8"
)


# ---------------------------------------------------------
# 2. Split document into sentences
# ---------------------------------------------------------

sentences = [
    sentence.strip()
    for sentence in text.split(".")
    if sentence.strip()
]


# ---------------------------------------------------------
# 3. Load embedding model
# ---------------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# 4. Create embeddings
# ---------------------------------------------------------

embeddings = model.encode(
    sentences
)


# ---------------------------------------------------------
# 5. Compare neighboring sentences
# ---------------------------------------------------------

# print("\n========== SEMANTIC SIMILARITY ==========\n")

# for i in range(len(sentences) - 1):

#     current = embeddings[i]
#     next_sentence = embeddings[i + 1]

#     similarity = cosine_similarity(
#         [current],
#         [next_sentence]
#     )[0][0]

#     print(
#         f"\nSentence {i + 1}:"
#     )

#     print(sentences[i])

#     print(
#         f"\nSentence {i + 2}:"
#     )

#     print(sentences[i + 1])

#     print(
#         f"\nSimilarity: {similarity:.3f}"
#     )

#     print("----------------------------------------")




# ---------------------------------------------------------
# 5. Build semantic chunks
# ---------------------------------------------------------

threshold = 0.50

chunks = []

current_chunk = [sentences[0]]


for i in range(len(sentences) - 1):

    similarity = cosine_similarity(
        [embeddings[i]],
        [embeddings[i + 1]]
    )[0][0]

    if similarity >= threshold:

        current_chunk.append(
            sentences[i + 1]
        )

    else:

        chunks.append(
            " ".join(current_chunk)
        )

        current_chunk = [
            sentences[i + 1]
        ]


# Add final chunk
if current_chunk:

    chunks.append(
        " ".join(current_chunk)
    )


# ---------------------------------------------------------
# 6. Display semantic chunks
# ---------------------------------------------------------

print("\n========== SEMANTIC CHUNKS ==========\n")

for i, chunk in enumerate(chunks, start=1):

    print(
        f"\n----- CHUNK {i} -----"
    )

    print(chunk)