from sentence_transformers import SentenceTransformer
import numpy as np


# Load the embedding model
model = SentenceTransformer("all-mpnet-base-v2")


# Original document
document = {
    "title": "Home Loan Policy",
    "section": "Required Documents"
}


# Small chunks created from the document
chunks = [
    "Applicants must submit identity proof and address proof.",
    "Applicants should provide salary slips and bank statements.",
    "Additional income documents may be required for self-employed applicants."
]


# Add document context to every chunk before embedding
contextual_chunks = []

for chunk in chunks:
    contextual_text = (
        f"Document: {document['title']}\n"
        f"Section: {document['section']}\n"
        f"Content: {chunk}"
    )

    contextual_chunks.append(contextual_text)


# Create embeddings from contextualized chunks
chunk_embeddings = model.encode(
    contextual_chunks,
    normalize_embeddings=True
)


# Search the contextualized chunks
def search_chunks(query, top_k=3):
    """Find chunks that are most similar to the user query."""

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = np.dot(chunk_embeddings, query_embedding)

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "index": index,
            "score": scores[index]
        }
        for index in top_indices
    ]


# User query
query = "What paperwork is needed for a home loan?"


print("=" * 70)
print("CONTEXTUALIZED CHUNKS")
print("=" * 70)

for i, chunk in enumerate(contextual_chunks):
    print(f"\nChunk ID: {i}")
    print(chunk)


print("\n" + "=" * 70)
print("USER QUERY")
print("=" * 70)

print(query)


# Retrieve relevant chunks
results = search_chunks(query)


print("\n" + "=" * 70)
print("RETRIEVAL RESULTS")
print("=" * 70)

for rank, result in enumerate(results, 1):

    index = result["index"]

    print(f"\nRank: {rank}")
    print(f"Chunk ID: {index}")
    print(f"Score: {result['score']:.4f}")
    print(f"Document: {chunks[index]}")