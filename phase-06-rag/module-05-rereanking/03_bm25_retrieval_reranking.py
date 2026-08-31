from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# Loads the Cross-Encoder model used to rerank retrieved documents.
def load_reranker():
    return CrossEncoder(RERANKER_MODEL_NAME)


# Converts documents into tokens so BM25 can perform keyword-based retrieval.
def tokenize_documents(documents):
    return [document.lower().split() for document in documents]


# Retrieves the top-K documents using BM25 keyword matching.
def bm25_retrieve(bm25, documents, query, top_k):
    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    ranked_ids = scores.argsort()[::-1][:top_k]

    results = []

    for document_id in ranked_ids:
        results.append({
            "id": int(document_id),
            "document": documents[document_id],
            "bm25_score": float(scores[document_id])
        })

    return results


# Reranks the BM25 candidates using the Cross-Encoder.
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
            "bm25_score": candidate["bm25_score"],
            "reranker_score": float(score)
        })

    results.sort(
        key=lambda result: result["reranker_score"],
        reverse=True
    )

    return results


# Prints the documents returned by BM25 before reranking.
def print_bm25_results(results):
    print("\nBM25 Retrieval Results")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"ID: {result['id']}")
        print(f"BM25 Score: {result['bm25_score']:.4f}")
        print(f"Document: {result['document']}")


# Prints the final Cross-Encoder ranking.
def print_reranked_results(results):
    print("\nCross-Encoder Reranked Results")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"ID: {result['id']}")
        print(f"BM25 Score: {result['bm25_score']:.4f}")
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
    print("BUILDING BM25 RETRIEVER")
    print("=" * 70)

    tokenized_documents = tokenize_documents(documents)

    bm25 = BM25Okapi(tokenized_documents)

    reranker = load_reranker()

    top_k = 5

    candidates = bm25_retrieve(
        bm25,
        documents,
        query,
        top_k
    )

    print(f"\nQuery: {query}")

    print_bm25_results(candidates)

    reranked_results = rerank_documents(
        reranker,
        query,
        candidates
    )

    print_reranked_results(reranked_results)


if __name__ == "__main__":
    main()