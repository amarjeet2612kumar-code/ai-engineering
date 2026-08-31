from sentence_transformers import SentenceTransformer
import numpy as np


# Create embeddings for text
model = SentenceTransformer("all-mpnet-base-v2")


# Parent documents
parents = {
    "P1": """
    Home Loan Eligibility

    Home loan eligibility depends on income, credit history,
    employment stability and existing financial liabilities.
    Applicants must satisfy the bank's eligibility criteria.
    """,

    "P2": """
    Home Loan Documents

    Home loan applicants need identity proof, address proof,
    income documents, salary slips and bank statements.
    """,

    "P3": """
    Personal Loan

    Personal loans can be used for various personal expenses.
    The loan amount depends on the applicant's income and credit profile.
    """
}


# Small child chunks mapped to their parent
children = [
    {
        "id": "C1",
        "parent_id": "P1",
        "text": "Home loan eligibility depends on the applicant's income."
    },
    {
        "id": "C2",
        "parent_id": "P1",
        "text": "Home loan applicants should have a good credit history."
    },
    {
        "id": "C3",
        "parent_id": "P1",
        "text": "Employment stability is considered during home loan evaluation."
    },
    {
        "id": "C4",
        "parent_id": "P2",
        "text": "Home loan applicants need identity proof and address proof."
    },
    {
        "id": "C5",
        "parent_id": "P2",
        "text": "Applicants should submit salary slips and bank statements."
    },
    {
        "id": "C6",
        "parent_id": "P3",
        "text": "Personal loans can be used for various personal expenses."
    }
]


# Create embeddings for child chunks only
child_texts = [child["text"] for child in children]
child_embeddings = model.encode(child_texts, normalize_embeddings=True)


# Search for the most relevant child chunk
def search_children(query, top_k=2):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = np.dot(child_embeddings, query_embedding)

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "child": children[i],
            "score": scores[i]
        }
        for i in top_indices
    ]


# Retrieve the parent using the parent ID stored with the child
def get_parent(parent_id):
    return parents[parent_id]


# User query
query = "What credit score is required for a home loan?"

print("=" * 70)
print("USER QUERY")
print("=" * 70)
print(query)


results = search_children(query)


print("\n" + "=" * 70)
print("CHILD RETRIEVAL")
print("=" * 70)

for rank, result in enumerate(results, 1):
    child = result["child"]

    print(f"\nRank: {rank}")
    print(f"Child ID: {child['id']}")
    print(f"Parent ID: {child['parent_id']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Child: {child['text']}")


print("\n" + "=" * 70)
print("PARENT CONTEXT")
print("=" * 70)

parent_ids = []

for result in results:
    parent_id = result["child"]["parent_id"]

    if parent_id not in parent_ids:
        parent_ids.append(parent_id)

        parent = get_parent(parent_id)

        print(f"\nParent ID: {parent_id}")
        print(parent)