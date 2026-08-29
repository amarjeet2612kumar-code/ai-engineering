import faiss
import numpy as np
import time


# -----------------------------------
# 1. Configuration
# -----------------------------------

np.random.seed(42)

num_vectors = 100_000
dimension = 128
k = 10


# -----------------------------------
# 2. Create dataset ONCE
# -----------------------------------

vectors = np.random.random(
    (num_vectors, dimension)
).astype("float32")


# -----------------------------------
# 3. Create query ONCE
# -----------------------------------

query_vector = np.random.random(
    (1, dimension)
).astype("float32")


# -----------------------------------
# 4. Exact Search
# -----------------------------------

exact_index = faiss.IndexFlatL2(dimension)

exact_index.add(vectors)

start = time.perf_counter()

exact_distances, exact_indices = exact_index.search(
    query_vector,
    k
)

exact_time = time.perf_counter() - start


print("\n===== Exact Search =====")
print("Time:", exact_time, "seconds")
print("IDs:", exact_indices[0])


# -----------------------------------
# 5. HNSW Index
# -----------------------------------

M = 16

hnsw_index = faiss.IndexHNSWFlat(
    dimension,
    M
)

hnsw_index.add(vectors)


# -----------------------------------
# 6. Test different efSearch values
# -----------------------------------

for ef_search in [32, 64, 128]:

    hnsw_index.hnsw.efSearch = ef_search

    start = time.perf_counter()

    hnsw_distances, hnsw_indices = hnsw_index.search(
        query_vector,
        k
    )

    hnsw_time = time.perf_counter() - start


    # -----------------------------------
    # Calculate Recall@10
    # -----------------------------------

    exact_set = set(exact_indices[0])

    hnsw_set = set(hnsw_indices[0])

    common_results = exact_set.intersection(
        hnsw_set
    )

    recall = len(common_results) / k


    # -----------------------------------
    # Calculate Speedup
    # -----------------------------------

    speedup = exact_time / hnsw_time


    # -----------------------------------
    # Print results
    # -----------------------------------

    print(f"\n===== HNSW efSearch = {ef_search} =====")

    print("Time:", hnsw_time, "seconds")

    print("IDs:", hnsw_indices[0])

    print("Recall@10:", recall)

    print("Speedup:", speedup)