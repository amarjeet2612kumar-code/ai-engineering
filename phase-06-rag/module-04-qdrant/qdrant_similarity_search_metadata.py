from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


# -----------------------------------
# 1. Connect to Qdrant
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
# 4. Metadata filter
# -----------------------------------

query_filter = Filter(
    must=[
        FieldCondition(
            key="department",
            match=MatchValue(value="Loans")
        )
    ]
)


# -----------------------------------
# 5. Similarity search + filter
# -----------------------------------

results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    query_filter=query_filter,
    limit=2
)


# -----------------------------------
# 6. Display results
# -----------------------------------

print("===== Filtered Search Results =====")

for result in results.points:

    print("\nID:", result.id)
    print("Score:", result.score)
    print("Payload:", result.payload)