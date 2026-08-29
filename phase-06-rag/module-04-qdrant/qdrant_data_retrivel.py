from qdrant_client import QdrantClient


# -----------------------------------
# 1. Connect to local Qdrant
# -----------------------------------

client = QdrantClient(path="./qdrant_data")


# -----------------------------------
# 2. Collection name
# -----------------------------------

collection_name = "bank_documents"


# -----------------------------------
# 3. Retrieve Point with ID = 1
# -----------------------------------

points = client.retrieve(
    collection_name=collection_name,
    ids=[1]
)


# -----------------------------------
# 4. Display retrieved Point
# -----------------------------------

for point in points:

    print("===== Retrieved Point =====")

    print("ID:", point.id)

    print("Vector:", point.vector)

    print("Payload:", point.payload)