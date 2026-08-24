from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# 1. Load embedding model
# ---------------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# 2. Text
# ---------------------------------------------------------

text = "Spark executor memory failure"


# ---------------------------------------------------------
# 3. Generate embedding
# ---------------------------------------------------------

embedding = model.encode(text)


# ---------------------------------------------------------
# 4. Inspect embedding
# ---------------------------------------------------------

print("\n========== EMBEDDING ==========\n")

print(embedding)

print("\n========== EMBEDDING INFO ==========\n")

print("Type      :", type(embedding))
print("Dimension :", len(embedding))