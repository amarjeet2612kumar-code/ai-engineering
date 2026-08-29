import ollama

response = ollama.embed(
    model="nomic-embed-text",
    input="How do I reset my banking password?"
)

embedding = response["embeddings"][0]

print("Embedding dimensions:", len(embedding))
print("First 10 values:", embedding[:10])