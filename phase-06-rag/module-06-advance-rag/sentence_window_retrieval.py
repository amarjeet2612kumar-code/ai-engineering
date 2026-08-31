from sentence_transformers import SentenceTransformer
import numpy as np


# Load the embedding model
model = SentenceTransformer("all-mpnet-base-v2")


# Document split into sentences
sentences = [
    "Home loan applicants must meet the bank's eligibility criteria.",
    "Applicants should have a stable source of income.",
    "A good credit history is required for home loan approval.",
    "Applicants must submit the required documents.",
    "These documents include identity proof, address proof and income documents.",
    "The bank may also request additional documents depending on the applicant's profile."
]


# Create embeddings for every sentence
sentence_embeddings = model.encode(
    sentences,
    normalize_embeddings=True
)


# Find the most relevant sentence
def search_sentence(query):
    """Find the sentence most similar to the user query."""

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = np.dot(sentence_embeddings, query_embedding)

    best_index = np.argmax(scores)

    return best_index, scores[best_index]


# Get nearby sentences around the matched sentence
def get_sentence_window(index, window_size=1):
    """Return the matched sentence plus nearby sentences."""

    start = max(0, index - window_size)
    end = min(len(sentences), index + window_size + 1)

    return sentences[start:end]


# User query
query = "What documents do I need for a home loan?"


print("=" * 70)
print("USER QUERY")
print("=" * 70)
print(query)


# Retrieve the most relevant sentence
best_index, best_score = search_sentence(query)


print("\n" + "=" * 70)
print("RETRIEVED SENTENCE")
print("=" * 70)

print(f"Sentence ID: {best_index}")
print(f"Score: {best_score:.4f}")
print(f"Sentence: {sentences[best_index]}")


# Expand the result using surrounding sentences
window = get_sentence_window(
    best_index,
    window_size=1
)


print("\n" + "=" * 70)
print("SENTENCE WINDOW")
print("=" * 70)

for sentence in window:
    print(sentence)