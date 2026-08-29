import faiss
import numpy as np


# -----------------------------------
# 1. Configuration
# -----------------------------------

np.random.seed(42)

num_vectors = 10_000
dimension = 128

k = 5


# -----------------------------------
# 2. Create vectors
# -----------------------------------

vectors = np.random.random(
    (num_vectors, dimension)
).astype("float32")


# -----------------------------------
# 3. Create query
# -----------------------------------

query_vector = np.random.random(
    (1, dimension)
).astype("float32")


# -----------------------------------
# 4. Exact Search
# -----------------------------------

exact_index = faiss.IndexFlatL2(dimension)

exact_index.add(vectors)

exact_distances, exact_indices = exact_index.search(
    query_vector,
    k
)


print("\n===== Exact Search =====")

print("IDs:", exact_indices[0])

print("Distances:", exact_distances[0])


# -----------------------------------
# 5. Create PQ index
# -----------------------------------

M = 16
nbits = 8

pq_index = faiss.IndexPQ(
    dimension,
    M,
    nbits
)


# -----------------------------------
# 6. Train PQ
# -----------------------------------

print("\nTraining PQ...")

pq_index.train(vectors)

print("Training completed.")

print("Is trained:", pq_index.is_trained)


# -----------------------------------
# 7. Add vectors
# -----------------------------------

pq_index.add(vectors)

print("Number of vectors:", pq_index.ntotal)


# -----------------------------------
# 8. PQ Search
# -----------------------------------

pq_distances, pq_indices = pq_index.search(
    query_vector,
    k
)


print("\n===== PQ Search =====")

print("IDs:", pq_indices[0])

print("Distances:", pq_distances[0])


# -----------------------------------
# 9. Recall@K
# -----------------------------------

exact_set = set(exact_indices[0])

pq_set = set(pq_indices[0])

common_results = exact_set.intersection(
    pq_set
)

recall = len(common_results) / k


print("\n===== Comparison =====")

print("Common IDs:", common_results)

print("Recall@5:", recall)