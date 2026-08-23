from pathlib import Path


# ---------------------------------------------------------
# 1. Load the document
# ---------------------------------------------------------

document_path = Path(
    "phase-06-rag/module-01-document-processing/"
    "documents/spark_troubleshooting.txt"
)

text = document_path.read_text(
    encoding="utf-8"
)


# ---------------------------------------------------------
# 2. Basic chunking configuration
# ---------------------------------------------------------

chunk_size = 300


# ---------------------------------------------------------
# 3. Split text into fixed-size chunks
# ---------------------------------------------------------

chunks = []

for start in range(0, len(text), chunk_size):

    chunk = text[start:start + chunk_size]

    chunks.append(chunk)


# ---------------------------------------------------------
# 4. Display chunks
# ---------------------------------------------------------

print("\n========== CHUNKS ==========\n")

for i, chunk in enumerate(chunks, start=1):

    print(f"\n----- CHUNK {i} -----")

    print(chunk)


# ---------------------------------------------------------
# 5. Statistics
# ---------------------------------------------------------

print("\n========== STATISTICS ==========\n")

print("Original characters :", len(text))
print("Chunk size          :", chunk_size)
print("Number of chunks    :", len(chunks))