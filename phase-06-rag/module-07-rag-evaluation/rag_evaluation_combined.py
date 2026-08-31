import ollama
import json


MODEL = "llama3.2:3b"


query = "What documents are required for a home loan?"


retrieved_context = [
    "Home loan applicants need identity proof and address proof.",
    "Applicants should submit salary slips and bank statements.",
    "Home loan eligibility depends on income and credit history.",
]


answer = """
Applicants need identity proof, address proof,
salary slips and bank statements.
"""


# ---------------------------------------------------------
# 1. Context Precision
# ---------------------------------------------------------

relevant_documents = {
    1: True,
    2: True,
    3: False,
}


relevant_count = 0
precision_sum = 0.0

for rank in range(1, len(retrieved_context) + 1):

    if relevant_documents[rank]:

        relevant_count += 1

        precision_at_rank = relevant_count / rank
        precision_sum += precision_at_rank


context_precision = precision_sum / relevant_count


# ---------------------------------------------------------
# 2. Context Recall
# ---------------------------------------------------------

required_information = [
    "identity proof",
    "address proof",
    "salary slips",
    "bank statements",
]


context_text = " ".join(retrieved_context).lower()

found_information = []

for item in required_information:

    if item in context_text:
        found_information.append(item)


context_recall = (
    len(found_information) / len(required_information)
)


# ---------------------------------------------------------
# 3. Answer Relevance
# ---------------------------------------------------------

relevance_prompt = f"""
Evaluate whether the answer directly addresses the question.

Question:
{query}

Answer:
{answer}

Give a score from 0 to 1.

1.0 = completely relevant
0.5 = partially relevant
0.0 = not relevant

Return ONLY JSON:

{{
    "score": 0.0,
    "reason": "short explanation"
}}
"""


relevance_response = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": relevance_prompt
        }
    ]
)


relevance_result = json.loads(
    relevance_response["message"]["content"]
)


# ---------------------------------------------------------
# 4. Faithfulness
# ---------------------------------------------------------

faithfulness_prompt = f"""
Evaluate whether the answer is completely supported
by the retrieved context.

Retrieved Context:
{context_text}

Answer:
{answer}

Give a score from 0 to 1.

1.0 = completely supported
0.5 = partially supported
0.0 = unsupported

Return ONLY JSON:

{{
    "score": 0.0,
    "reason": "short explanation"
}}
"""


faithfulness_response = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": faithfulness_prompt
        }
    ]
)


faithfulness_result = json.loads(
    faithfulness_response["message"]["content"]
)


# ---------------------------------------------------------
# Print Results
# ---------------------------------------------------------

print("=" * 70)
print("COMBINED RAG EVALUATION")
print("=" * 70)

print()
print("USER QUERY")
print(query)

print()
print("GENERATED ANSWER")
print(answer)

print()
print("=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

print(f"Context Precision : {context_precision:.4f}")
print(f"Context Recall    : {context_recall:.4f}")

print(
    f"Answer Relevance  : "
    f"{relevance_result['score']:.4f}"
)

print(
    f"Faithfulness      : "
    f"{faithfulness_result['score']:.4f}"
)

print()
print("Answer Relevance Reason:")
print(relevance_result["reason"])

print()
print("Faithfulness Reason:")
print(faithfulness_result["reason"])