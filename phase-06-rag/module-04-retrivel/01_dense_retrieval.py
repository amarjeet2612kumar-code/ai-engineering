# ============================================================
# Practical 1: Dense Retrieval using Cosine Similarity
# ============================================================

import ollama
import numpy as np
import faiss


# ------------------------------------------------------------
# 1. Documents
# ------------------------------------------------------------

documents = [
    "Home loan eligibility requires sufficient income and a good credit history.",

    "Home loan applicants need identity proof, address proof and income documents.",

    "Personal loans can be used for various personal expenses.",

    "Home loan interest rates depend on the applicant's profile and loan tenure.",

    "Credit cards provide a revolving credit facility to customers."
]


# ------------------------------------------------------------
# 2. Embedding Model
# ------------------------------------------------------------

EMBEDDING_MODEL = "nomic-embed-text"


# ------------------------------------------------------------
# 3. Function to create embedding
# ------------------------------------------------------------

def create_embedding(text):

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response["embeddings"][0]


# ------------------------------------------------------------
# 4. Create embeddings for all documents
# ------------------------------------------------------------

print("Creating document embeddings...")

document_embeddings = []

for document in documents:

    embedding = create_embedding(document)

    document_embeddings.append(embedding)


# Convert Python list → NumPy array
document_embeddings = np.array(
    document_embeddings,
    dtype="float32"
)


print(
    "Document embedding shape:",
    document_embeddings.shape
)


# ------------------------------------------------------------
# 5. Normalize document vectors
# ------------------------------------------------------------
#
# After normalization:
#
#       Inner Product = Cosine Similarity
#
# ------------------------------------------------------------

faiss.normalize_L2(document_embeddings)


# ------------------------------------------------------------
# 6. Create FAISS index
# ------------------------------------------------------------

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)


print(
    "Vector dimension:",
    dimension
)


# ------------------------------------------------------------
# 7. Add document vectors to FAISS
# ------------------------------------------------------------

index.add(document_embeddings)


print(
    "Number of vectors in index:",
    index.ntotal
)


# ------------------------------------------------------------
# 8. User Query
# ------------------------------------------------------------

query = "What documents are required for a home loan?"

print("\nQuery:")
print(query)


# ------------------------------------------------------------
# 9. Create query embedding
# ------------------------------------------------------------

query_embedding = create_embedding(query)


# ------------------------------------------------------------
# 10. Convert query embedding to NumPy
# ------------------------------------------------------------

query_vector = np.array(
    [query_embedding],
    dtype="float32"
)


print(
    "\nQuery vector shape:",
    query_vector.shape
)


# ------------------------------------------------------------
# 11. Normalize query vector
# ------------------------------------------------------------

faiss.normalize_L2(query_vector)


# ------------------------------------------------------------
# 12. Search
# ------------------------------------------------------------

k = 3

similarities, indices = index.search(
    query_vector,
    k
)


# ------------------------------------------------------------
# 13. Display raw search output
# ------------------------------------------------------------

print("\nRaw Search Output")

print("Similarities:")
print(similarities)

print("\nIndices:")
print(indices)


# ------------------------------------------------------------
# 14. Map IDs back to documents
# ------------------------------------------------------------

print("\nTop Results:")

for rank, (similarity, index_id) in enumerate(
    zip(similarities[0], indices[0]),
    start=1
):

    print(
        f"\nRank: {rank}"
    )

    print(
        f"ID: {index_id}"
    )

    print(
        f"Cosine Similarity: {similarity:.4f}"
    )

    print(
        f"Document: {documents[index_id]}"
    )