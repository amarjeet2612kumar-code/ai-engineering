from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


client = QdrantClient(path="./qdrant_data")


# Create collection
client.create_collection(
    collection_name="bank_documents",
    vectors_config=VectorParams(
        size=4,
        distance=Distance.COSINE
    )
)


# Create points
points = [
    PointStruct(
        id=1,
        vector=[0.10, 0.20, 0.30, 0.40],
        payload={
            "department": "Loans",
            "document": "home_loan_policy.pdf",
            "text": "Home loan eligibility requirements"
        }
    ),

    PointStruct(
        id=2,
        vector=[0.80, 0.70, 0.60, 0.50],
        payload={
            "department": "Credit Card",
            "document": "credit_card_policy.pdf",
            "text": "Credit card eligibility requirements"
        }
    ),

    PointStruct(
        id=3,
        vector=[0.12, 0.22, 0.32, 0.42],
        payload={
            "department": "Loans",
            "document": "personal_loan_policy.pdf",
            "text": "Personal loan eligibility requirements"
        }
    )
]


# Insert points
client.upsert(
    collection_name="bank_documents",
    points=points
)


print("Points inserted successfully.")

