import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# Loads the embedding model used for semantic retrieval.
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# Loads the Cross-Encoder used for final reranking.
def load_reranker():
    return CrossEncoder(RERANKER_MODEL_NAME)


# Creates a FAISS index containing normalized document embeddings.
def build_dense_index(model, documents):
    embeddings = model.encode(
        documents,
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


# Retrieves documents using dense cosine similarity.
def dense_retrieve(model, index, documents, query, top_k):
    query_vector = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_vector)

    scores, ids = index.search(query_vector, top_k)

    results = []

    for score, document_id in zip(scores[0], ids[0]):
        results.append({
            "id": int(document_id),
            "score": float(score)
        })

    return results


# Creates the BM25 index from tokenized documents.
def build_bm25_index(documents):
    tokenized_documents = [
        document.lower().split()
        for document in documents
    ]

    return BM25Okapi(tokenized_documents)


# Retrieves documents using BM25 keyword matching.
def bm25_retrieve(bm25, query, top_k):
    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    ids = scores.argsort()[::-1][:top_k]

    results = []

    for document_id in ids:
        results.append({
            "id": int(document_id),
            "score": float(scores[document_id])
        })

    return results


# Combines Dense and BM25 rankings using Reciprocal Rank Fusion.
def rrf_fusion(dense_results, bm25_results, k=60):
    rrf_scores = {}

    for rank, result in enumerate(dense_results, start=1):
        document_id = result["id"]

        rrf_scores[document_id] = (
            rrf_scores.get(document_id, 0)
            + 1 / (k + rank)
        )

    for rank, result in enumerate(bm25_results, start=1):
        document_id = result["id"]

        rrf_scores[document_id] = (
            rrf_scores.get(document_id, 0)
            + 1 / (k + rank)
        )

    results = []

    for document_id, score in rrf_scores.items():
        results.append({
            "id": document_id,
            "rrf_score": score
        })

    results.sort(
        key=lambda result: result["rrf_score"],
        reverse=True
    )

    return results


# Reranks the RRF candidates using the Cross-Encoder.
def rerank_documents(reranker, query, rrf_results, documents):
    pairs = [
        [query, documents[result["id"]]]
        for result in rrf_results
    ]

    scores = reranker.predict(pairs)

    results = []

    for result, score in zip(rrf_results, scores):
        results.append({
            "id": result["id"],
            "rrf_score": result["rrf_score"],
            "reranker_score": float(score)
        })

    results.sort(
        key=lambda result: result["reranker_score"],
        reverse=True
    )

    return results


# Prints the ranking before reranking.
def print_rrf_results(results, documents):
    print("\nBEFORE RERANKING — RRF")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        print(
            f"Rank: {rank} | "
            f"ID: {result['id']} | "
            f"RRF: {result['rrf_score']:.6f}"
        )

        print(f"Document: {documents[result['id']]}")
        print()


# Prints the final ranking after Cross-Encoder reranking.
def print_reranked_results(results, documents):
    print("\nAFTER RERANKING — CROSS-ENCODER")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        print(
            f"Rank: {rank} | "
            f"ID: {result['id']} | "
            f"RRF: {result['rrf_score']:.6f} | "
            f"Reranker: {result['reranker_score']:.4f}"
        )

        print(f"Document: {documents[result['id']]}")
        print()


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
    print("BEFORE VS AFTER RERANKING")
    print("=" * 70)

    embedding_model = load_embedding_model()
    reranker = load_reranker()

    dense_index = build_dense_index(
        embedding_model,
        documents
    )

    bm25 = build_bm25_index(documents)

    top_k = 5

    # Retrieves semantic candidates using Dense Retrieval.
    dense_results = dense_retrieve(
        embedding_model,
        dense_index,
        documents,
        query,
        top_k
    )

    # Retrieves keyword-based candidates using BM25.
    bm25_results = bm25_retrieve(
        bm25,
        query,
        top_k
    )

    # Combines both retrieval rankings using RRF.
    rrf_results = rrf_fusion(
        dense_results,
        bm25_results
    )

    print_rrf_results(
        rrf_results,
        documents
    )

    # Reranks the RRF candidates using the Cross-Encoder.
    reranked_results = rerank_documents(
        reranker,
        query,
        rrf_results,
        documents
    )

    print_reranked_results(
        reranked_results,
        documents
    )


if __name__ == "__main__":
    main()