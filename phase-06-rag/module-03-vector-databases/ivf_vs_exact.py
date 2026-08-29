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

nlist = 100


# -----------------------------------
# 2. Create dataset
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
# 5. Create IVF index
# -----------------------------------

quantizer = faiss.IndexFlatL2(dimension)

ivf_index = faiss.IndexIVFFlat(
    quantizer,
    dimension,
    nlist,
    faiss.METRIC_L2
)


# -----------------------------------
# 6. Train IVF
# -----------------------------------

print("\nTraining IVF...")

ivf_index.train(vectors)

print("Training completed.")

print("Is trained:", ivf_index.is_trained)


# -----------------------------------
# 7. Add vectors
# -----------------------------------

ivf_index.add(vectors)

print("Number of vectors:", ivf_index.ntotal)


# -----------------------------------
# 8. Test different nprobe values
# -----------------------------------

for nprobe in [1, 5, 10]:

    ivf_index.nprobe = nprobe

    start = time.perf_counter()

    ivf_distances, ivf_indices = ivf_index.search(
        query_vector,
        k
    )

    ivf_time = time.perf_counter() - start


    # -----------------------------------
    # Calculate Recall@10
    # -----------------------------------

    exact_set = set(exact_indices[0])

    ivf_set = set(ivf_indices[0])

    common_results = exact_set.intersection(
        ivf_set
    )

    recall = len(common_results) / k


    # -----------------------------------
    # Calculate Speedup
    # -----------------------------------

    speedup = exact_time / ivf_time


    # -----------------------------------
    # Print
    # -----------------------------------

    print(
        f"\n===== IVF nprobe = {nprobe} ====="
    )

    print("Time:", ivf_time, "seconds")

    print("IDs:", ivf_indices[0])

    print("Recall@10:", recall)

    print("Speedup:", speedup)