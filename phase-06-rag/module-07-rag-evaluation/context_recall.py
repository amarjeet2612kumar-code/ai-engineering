# context_recall.py

query = "What documents are required for a home loan?"

reference_answer = [
    "identity proof",
    "address proof",
    "income documents",
    "salary slips",
    "bank statements",
]

retrieved_documents = [
    "Home loan applicants need identity proof and income documents.",
    "Home loan applicants should submit salary slips.",
    "Home loan interest rates depend on the applicant's profile and loan tenure.",
]


print("=" * 70)
print("CONTEXT RECALL EVALUATION")
print("=" * 70)

print()
print("USER QUERY")
print("=" * 70)
print(query)

print()
print("REQUIRED INFORMATION")
print("=" * 70)

for item in reference_answer:
    print(item)

print()
print("RETRIEVED DOCUMENTS")
print("=" * 70)

for document in retrieved_documents:
    print(document)


# ---------------------------------------------------------
# Check which required information was retrieved
# ---------------------------------------------------------

retrieved_text = " ".join(retrieved_documents).lower()

found = []
missing = []

for item in reference_answer:

    if item.lower() in retrieved_text:
        found.append(item)
    else:
        missing.append(item)


# ---------------------------------------------------------
# Calculate Context Recall
# ---------------------------------------------------------

recall = len(found) / len(reference_answer)


print()
print("=" * 70)
print("CONTEXT RECALL")
print("=" * 70)

print()
print("FOUND:")
for item in found:
    print(f"✓ {item}")

print()
print("MISSING:")
for item in missing:
    print(f"✗ {item}")

print()
print(f"Required information: {len(reference_answer)}")
print(f"Retrieved information: {len(found)}")
print(f"Context Recall: {recall:.4f}")