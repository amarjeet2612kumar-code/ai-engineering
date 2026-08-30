# ============================================================
# Practical 2: Sparse Retrieval using BM25
# ============================================================

import re
from rank_bm25 import BM25Okapi


# ------------------------------------------------------------
# 1. Documents
# ------------------------------------------------------------

documents = [
    "Home loan eligibility requires sufficient income and a good credit history.",

    "Home loan applicants need identity proof, address proof and income documents.",

    "Personal loans can be used for various personal expenses.",

    "Home loan interest rates depend on the applicant's profile and loan tenure.",

    "Credit cards provide a revolving credit facility to customers."
]


# ------------------------------------------------------------
# 2. Tokenization / Text Preprocessing
# ------------------------------------------------------------

def tokenize(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    # Split text into individual words
    return text.split()


# ------------------------------------------------------------
# 3. Tokenize all documents
# ------------------------------------------------------------

tokenized_documents = [
    tokenize(document)
    for document in documents
]


print("Tokenized Documents:\n")

for index, tokens in enumerate(tokenized_documents):

    print(
        f"ID {index}: {tokens}"
    )


# ------------------------------------------------------------
# 4. Create BM25 Index
# ------------------------------------------------------------

bm25 = BM25Okapi(
    tokenized_documents
)


# ------------------------------------------------------------
# 5. User Query
# ------------------------------------------------------------

query = "What documents are required for a home loan?"


print("\nQuery:")
print(query)


# ------------------------------------------------------------
# 6. Tokenize Query
# ------------------------------------------------------------

tokenized_query = tokenize(query)


print("\nTokenized Query:")
print(tokenized_query)


# ------------------------------------------------------------
# 7. Calculate BM25 Scores
# ------------------------------------------------------------

scores = bm25.get_scores(
    tokenized_query
)


# ------------------------------------------------------------
# 8. Display score for every document
# ------------------------------------------------------------

print("\nBM25 Scores:")

for index, score in enumerate(scores):

    print(
        f"ID: {index} | "
        f"Score: {score:.4f} | "
        f"Document: {documents[index]}"
    )


# ------------------------------------------------------------
# 9. Select Top-K documents
# ------------------------------------------------------------

k = 3


top_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True
)[:k]


# ------------------------------------------------------------
# 10. Display Top-K Results
# ------------------------------------------------------------

print("\nTop Results:")

for rank, index_id in enumerate(
    top_indices,
    start=1
):

    print(
        f"\nRank: {rank}"
    )

    print(
        f"ID: {index_id}"
    )

    print(
        f"BM25 Score: {scores[index_id]:.4f}"
    )

    print(
        f"Document: {documents[index_id]}"
    )

