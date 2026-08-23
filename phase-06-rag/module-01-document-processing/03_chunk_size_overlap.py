from pathlib import Path


# ---------------------------------------------------------
# 1. Load document
# ---------------------------------------------------------

document_path = Path(
    "phase-06-rag/module-01-document-processing/"
    "documents/spark_troubleshooting.txt"
)

text = document_path.read_text(
    encoding="utf-8"
)


# ---------------------------------------------------------
# 2. Basic fixed-size chunker
# ---------------------------------------------------------

def create_chunks(text, chunk_size, overlap=0):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------
# 3. Compare different chunk sizes
# ---------------------------------------------------------

chunk_sizes = [100, 300, 600]


for size in chunk_sizes:

    chunks = create_chunks(
        text=text,
        chunk_size=size,
        overlap=50
    )

    print("\n================================")
    print(f"CHUNK SIZE = {size}")
    print("================================")

    print("Number of chunks:", len(chunks))

    for i, chunk in enumerate(chunks, start=1):

        print(
            f"\n--- Chunk {i} "
            f"({len(chunk)} chars) ---"
        )

        print(chunk)