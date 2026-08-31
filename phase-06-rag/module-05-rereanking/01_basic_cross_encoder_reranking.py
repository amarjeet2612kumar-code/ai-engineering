from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# Loads the Cross-Encoder model used to score query-document relevance.
def load_reranker():
    return CrossEncoder(MODEL_NAME)


# Scores every document against the user's query.
def rerank_documents(model, query, documents):
    pairs = [[query, document] for document in documents]

    scores = model.predict(pairs)

    results = []

    for document, score in zip(documents, scores):
        results.append({
            "document": document,
            "score": float(score)
        })

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results


# Prints the documents in their new relevance order.
def print_results(results):
    print("\nReranked Results")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Document: {result['document']}")


def main():
    query = "What documents are required for a home loan?"

    documents = [
        "Home loan applicants need identity proof, address proof and income documents.",
        "Home loan interest rates depend on the applicant's profile and loan tenure.",
        "Credit cards provide a revolving credit facility to customers.",
        "Home loan eligibility requires sufficient income and a good credit history.",
        "Personal loans can be used for various personal expenses."
    ]

    model = load_reranker()

    results = rerank_documents(
        model,
        query,
        documents
    )

    print_results(results)


if __name__ == "__main__":
    main()