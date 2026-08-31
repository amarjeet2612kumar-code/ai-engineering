import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# Loads the embedding model used for dense retrieval.
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# Loads the Cross-Encoder used for final reranking.
def load_reranker():
    return CrossEncoder(RERANKER_MODEL_NAME)


# Creates a FAISS index from normalized document embeddings.
def build_dense_index(model, documents):
    embeddings = model.encode(
        documents,
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


# Retrieves candidate documents using cosine similarity.
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
            "document": documents[document_id],
            "score": float(score)
        })

    return results


# Creates a BM25 index from tokenized documents.
def build_bm25_index(documents):
    tokenized_documents = [
        document.lower().split()
        for document in documents
    ]

    return BM25Okapi(tokenized_documents)


# Retrieves candidate documents using BM25.
def bm25_retrieve(bm25, documents, query, top_k):
    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    ranked_ids = scores.argsort()[::-1][:top_k]

    results = []

    for document_id in ranked_ids:
        results.append({
            "id": int(document_id),
            "document": documents[document_id],
            "score": float(scores[document_id])
        })

    return results


# Combines dense and BM25 rankings using Reciprocal Rank Fusion.
def rrf_fusion(dense_results, bm25_results, k=60):
    scores = {}

    for rank, result in enumerate(dense_results, start=1):
        document_id = result["id"]

        if document_id not in scores:
            scores[document_id] = 0

        scores[document_id] += 1 / (k + rank)

    for rank, result in enumerate(bm25_results, start=1):
        document_id = result["id"]

        if document_id not in scores:
            scores[document_id] = 0

        scores[document_id] += 1 / (k + rank)

    results = []

    for document_id, score in scores.items():
        results.append({
            "id": document_id,
            "rrf_score": score
        })

    results.sort(
        key=lambda result: result["rrf_score"],
        reverse=True
    )

    return results


# Adds document text to the documents produced by RRF.
def add_documents(rrf_results, documents):
    results = []

    for result in rrf_results:
        results.append({
            "id": result["id"],
            "document": documents[result["id"]],
            "rrf_score": result["rrf_score"]
        })

    return results


# Reranks the fused candidates using the Cross-Encoder.
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
            "rrf_score": candidate["rrf_score"],
            "reranker_score": float(score)
        })

    results.sort(
        key=lambda result: result["reranker_score"],
        reverse=True
    )

    return results


# Prints the final reranked results.
def print_results(results):
    print("\nFinal Reranked Results")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"ID: {result['id']}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
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
    print("BUILDING RETRIEVERS")
    print("=" * 70)

    embedding_model = load_embedding_model()
    reranker = load_reranker()

    dense_index = build_dense_index(
        embedding_model,
        documents
    )

    bm25 = build_bm25_index(documents)

    top_k = 5

    # First-stage dense retrieval.
    dense_results = dense_retrieve(
        embedding_model,
        dense_index,
        documents,
        query,
        top_k
    )

    # First-stage BM25 retrieval.
    bm25_results = bm25_retrieve(
        bm25,
        documents,
        query,
        top_k
    )

    # Combine both rankings using RRF.
    rrf_results = rrf_fusion(
        dense_results,
        bm25_results
    )

    rrf_results = add_documents(
        rrf_results,
        documents
    )

    # Rerank the fused candidates.
    final_results = rerank_documents(
        reranker,
        query,
        rrf_results
    )

    print(f"\nQuery: {query}")

    print_results(final_results)


if __name__ == "__main__":
    main()