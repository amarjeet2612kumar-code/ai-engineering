import numpy as np
import faiss


# -----------------------------------
# 1. Our documents
# -----------------------------------

documents = [
    "Python is easy to learn",
    "Python is used for data engineering",
    "Spark processes large datasets",
    "Kafka handles event streaming",
    "Machine learning uses data"
]


# -----------------------------------
# 2. Create vectors
# -----------------------------------

vectors = np.array([
    [1.0, 1.0],
    [1.2, 1.1],
    [5.0, 5.0],
    [8.0, 8.0],
    [6.0, 5.5]
], dtype="float32")


# -----------------------------------
# 3. Get vector dimension
# -----------------------------------

dimension = vectors.shape[1]

print("Vector dimension:", dimension)


# -----------------------------------
# 4. Create FAISS index
# -----------------------------------

index = faiss.IndexFlatL2(dimension)

print("row index:", index)

print("Vectors before adding:", index.ntotal)


# -----------------------------------
# 5. Add vectors to FAISS
# -----------------------------------

index.add(vectors)

print("Vectors after adding:", index.ntotal)


# -----------------------------------
# 6. Create query vector
# -----------------------------------

query_vector = np.array([
    [1.1, 1.0]
], dtype="float32")

print("Query vectors after adding:", query_vector)
print("Query vectors after adding2:", query_vector[0])
# -----------------------------------
# 7. Search
# -----------------------------------

k = 3

distances, indices = index.search(query_vector, k)


# -----------------------------------
# 8. Display raw FAISS results
# -----------------------------------

print("\nDistances:")
print(distances)

print("\nIndices:")
print(indices)


# -----------------------------------
# 9. Map IDs back to documents
# -----------------------------------

print("\nTop results:")

for distance, index_id in zip(distances[0], indices[0]):
    print(
        f"ID: {index_id} | "
        f"Distance: {distance:.4f} | "
        f"Document: {documents[index_id]}"
    )