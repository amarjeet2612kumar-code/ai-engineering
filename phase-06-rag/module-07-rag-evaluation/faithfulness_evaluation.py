import ollama


MODEL = "llama3.2:3b"


context = """
Home loan applicants need identity proof,
address proof and income documents.
"""

answer = """
Applicants need identity proof, address proof,
income documents and a minimum credit score of 750.
"""


prompt = f"""
You are evaluating the faithfulness of an answer.

Context:
{context}

Answer:
{answer}

Check whether the answer is completely supported by the context.

Return:
- SUPPORTED if every claim is supported
- UNSUPPORTED if any claim is not supported

Also give a short reason.
"""


response = ollama.chat(
    model=MODEL,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("Faithfulness Evaluation")
print("=" * 60)
print(response["message"]["content"])