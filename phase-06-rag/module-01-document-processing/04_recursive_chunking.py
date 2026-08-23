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
# 2. Recursive chunking function
# ---------------------------------------------------------

def recursive_split(
    text,
    chunk_size,
    separators
):

    # If text already fits, return it
    if len(text) <= chunk_size:
        return [text]

    # If no separators remain, force split
    if not separators:
        return [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size)
        ]

    separator = separators[0]

    parts = text.split(separator)

    chunks = []
    current = ""

    for part in parts:

        candidate = (
            current + separator + part
            if current
            else part
        )

        if len(candidate) <= chunk_size:

            current = candidate

        else:

            if current:
                chunks.append(current)

            # Recursively split oversized part
            if len(part) > chunk_size:

                smaller_chunks = recursive_split(
                    part,
                    chunk_size,
                    separators[1:]
                )

                chunks.extend(smaller_chunks)

                current = ""

            else:

                current = part

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------
# 3. Define hierarchy
# ---------------------------------------------------------

separators = [
    "\n\n",   # paragraph
    "\n",     # line
    ". ",     # sentence
    " ",      # word
]


# ---------------------------------------------------------
# 4. Create chunks
# ---------------------------------------------------------

chunks = recursive_split(
    text=text,
    chunk_size=300,
    separators=separators
)


# ---------------------------------------------------------
# 5. Display chunks
# ---------------------------------------------------------

print("\n========== RECURSIVE CHUNKS ==========\n")

for i, chunk in enumerate(chunks, start=1):

    print(
        f"\n----- CHUNK {i} "
        f"({len(chunk)} characters) -----"
    )

    print(chunk)