import numpy as np
import faiss
import ollama


# ---------------------------------------
# 1. Documents
# ---------------------------------------

documents = [
    "How do I reset my internet banking password?",
    "What should I do if I forget my banking credentials?",
    "How can I apply for a personal loan?",
    "What are the charges for ATM withdrawals?",
    "How do I block my debit card?",
    "How can I change my mobile banking PIN?",
    "What documents are required to open a savings account?",
]


# ---------------------------------------
# 2. Embedding function
# ---------------------------------------

def create_embedding(text):
    response = ollama.embed(
        model="nomic-embed-text",
        input=text
    )

    return response["embeddings"][0]


# ---------------------------------------
# 3. Create embeddings for documents
# ---------------------------------------

document_embeddings = []

for document in documents:
    embedding = create_embedding(document)
    document_embeddings.append(embedding)


# ---------------------------------------
# 4. Convert to NumPy array
# ---------------------------------------

vectors = np.array(
    document_embeddings,
    dtype="float32"
)


print("Number of documents:", len(documents))
print("Vector shape:", vectors.shape)


# ---------------------------------------
# 5. Create FAISS index
# ---------------------------------------

dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)


# ---------------------------------------
# 6. Add document vectors
# ---------------------------------------

index.add(vectors)

print("Vectors stored in FAISS:", index.ntotal)


# ---------------------------------------
# 7. User query
# ---------------------------------------

query = "I cannot remember my online banking login password"


# ---------------------------------------
# 8. Convert query into embedding
# ---------------------------------------

query_embedding = create_embedding(query)

query_vector = np.array(
    [query_embedding],
    dtype="float32"
)


# ---------------------------------------
# 9. Search
# ---------------------------------------

k = 3

distances, indices = index.search(
    query_vector,
    k
)


# ---------------------------------------
# 10. Display results
# ---------------------------------------

print("\nQuery:")
print(query)

print("\nSearch Results:")

for distance, index_id in zip(
    distances[0],
    indices[0]
):
    print(
        f"\nID: {index_id}"
        f"\nDistance: {distance:.4f}"
        f"\nDocument: {documents[index_id]}"
    )