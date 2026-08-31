import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


# Models used for dense retrieval and document reranking.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# Loads the embedding model used for initial candidate retrieval.
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# Loads the Cross-Encoder used to rerank retrieved documents.
def load_reranker_model():
    return CrossEncoder(RERANKER_MODEL_NAME)


# Creates a FAISS index containing normalized document embeddings.
def build_vector_index(embedding_model, documents):
    embeddings = embedding_model.encode(
        documents,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


# Retrieves the top-K documents using cosine similarity.
def dense_retrieve(embedding_model, index, documents, query, top_k):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_id in zip(scores[0], indices[0]):
        results.append({
            "id": int(index_id),
            "document": documents[index_id],
            "dense_score": float(score)
        })

    return results


# Reranks the dense-retrieved candidates using the Cross-Encoder.
def rerank_documents(reranker, query, candidates):
    pairs = [
        [query, candidate["document"]]
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    results = []

    for candidate, score in zip(candidates, scores):
        results.append({
            "id": candidate["id"],
            "document": candidate["document"],
            "dense_score": candidate["dense_score"],
            "reranker_score": float(score)
        })

    results.sort(
        key=lambda result: result["reranker_score"],
        reverse=True
    )

    return results


# Prints the initial dense ranking.
def print_dense_results(results):
    print("\nDense Retrieval Results")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"ID: {result['id']}")
        print(f"Dense Score: {result['dense_score']:.4f}")
        print(f"Document: {result['document']}")


# Prints the final reranked results.
def print_reranked_results(results):
    print("\nCross-Encoder Reranked Results")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"ID: {result['id']}")
        print(f"Dense Score: {result['dense_score']:.4f}")
        print(f"Reranker Score: {result['reranker_score']:.4f}")
        print(f"Document: {result['document']}")


def main():
    query = "What documents are required for a home loan?"

    documents = [
        "Home loan eligibility requires sufficient income and a good credit history.",
        "Home loan applicants need identity proof, address proof and income documents.",
        "Personal loans can be used for various personal expenses.",
        "Home loan interest rates depend on the applicant's profile and loan tenure.",
        "Credit cards provide a revolving credit facility to customers.",
        "Home loan applicants should submit salary slips and bank statements.",
        "Home loan approval depends on income, credit score and existing liabilities."
    ]

    print("=" * 70)
    print("BUILDING MODELS")
    print("=" * 70)

    embedding_model = load_embedding_model()
    reranker = load_reranker_model()

    print("\nBuilding vector index...")

    index = build_vector_index(
        embedding_model,
        documents
    )

    print(f"Number of vectors: {index.ntotal}")

    # Retrieve a larger candidate set before reranking.
    top_k = 5

    dense_results = dense_retrieve(
        embedding_model,
        index,
        documents,
        query,
        top_k
    )

    print(f"\nQuery: {query}")

    print_dense_results(dense_results)

    reranked_results = rerank_documents(
        reranker,
        query,
        dense_results
    )

    print_reranked_results(reranked_results)


if __name__ == "__main__":
    main()