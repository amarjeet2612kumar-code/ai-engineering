from qdrant_client import QdrantClient


# -----------------------------------
# 1. Connect to local Qdrant
# -----------------------------------

client = QdrantClient(path="./qdrant_data")


# -----------------------------------
# 2. Collection
# -----------------------------------

collection_name = "bank_documents"


# -----------------------------------
# 3. Query vector
# -----------------------------------

query_vector = [0.11, 0.21, 0.31, 0.41]


# -----------------------------------
# 4. Similarity search
# -----------------------------------

results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=2
)


# -----------------------------------
# 5. Display results
# -----------------------------------

print("===== Search Results =====")

for result in results.points:

    print("\nID:", result.id)
    print("Score:", result.score)
    print("Payload:", result.payload)