from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


# 1. Create local Qdrant client
client = QdrantClient(path="./qdrant_data")


# 2. Create collection
client.create_collection(
    collection_name="bank_documents",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)


# 3. Check collection
collection_info = client.get_collection("bank_documents")

print(collection_info)