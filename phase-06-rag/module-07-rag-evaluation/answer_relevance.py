import ollama
import json


MODEL = "llama3.2:3b"


query = "What documents are required for a home loan?"

answer = """
Applicants need identity proof, address proof,
income documents, salary slips and bank statements.
"""


prompt = f"""
You are an evaluator for a RAG system.

Your task is to evaluate whether the answer is relevant to
the user's question.

User Question:
{query}

Generated Answer:
{answer}

Evaluation rules:

1. The answer must directly address the user's question.
2. The answer should provide information needed to answer the question.
3. Do not judge whether the answer is factually correct.
4. Only judge relevance to the question.

Give a relevance score from 0 to 1:

1.0 = Completely relevant
0.5 = Partially relevant
0.0 = Not relevant

Return ONLY valid JSON:

{{
    "score": 0.0,
    "reason": "short explanation"
}}
"""


response = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


result = response["message"]["content"]

print("=" * 70)
print("ANSWER RELEVANCE EVALUATION")
print("=" * 70)

print()
print("USER QUERY")
print("=" * 70)
print(query)

print()
print("GENERATED ANSWER")
print("=" * 70)
print(answer)

print()
print("=" * 70)
print("LLM JUDGE RESULT")
print("=" * 70)

print(result)