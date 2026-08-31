import ollama


MODEL = "llama3.2:3b"


query = "What documents are required for a home loan?"


documents = [
    "Home loan applicants need identity proof and income documents.",
    "Home loan applicants should submit salary slips and bank statements.",
    "Home loan interest rates depend on the applicant's profile and loan tenure.",
    "Credit cards provide a revolving credit facility to customers.",
]


for rank, document in enumerate(documents, start=1):

    prompt = f"""
You are evaluating whether a retrieved document is relevant to a user query.

User Query:
{query}

Retrieved Document:
{document}

A document is RELEVANT if it contains information that can help answer
the user's question.

A document is NOT_RELEVANT if it does not help answer the question.

Examples:

Query: What documents are required for a home loan?
Document: Applicants need identity proof and income documents.
Answer: RELEVANT

Query: What documents are required for a home loan?
Document: Credit cards provide a revolving credit facility.
Answer: NOT_RELEVANT

Now evaluate the document above.

Return exactly one word:
RELEVANT
or
NOT_RELEVANT
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    result = response["message"]["content"].strip()

    print(f"Rank: {rank}")
    print(f"Document: {document}")
    print(f"Evaluation: {result}")
    print("-" * 60)