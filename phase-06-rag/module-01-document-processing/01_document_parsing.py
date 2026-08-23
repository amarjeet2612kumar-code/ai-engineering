from pathlib import Path


# ---------------------------------------------------------
# 1. Locate the document
# ---------------------------------------------------------

document_path = Path(
    "phase-06-rag/module-01-document-processing/documents/"
    "spark_troubleshooting.txt"
)


# ---------------------------------------------------------
# 2. Read the document
# ---------------------------------------------------------

text = document_path.read_text(
    encoding="utf-8"
)


# ---------------------------------------------------------
# 3. Display basic information
# ---------------------------------------------------------

print("\n========== DOCUMENT ==========\n")

print(text)


# ---------------------------------------------------------
# 4. Basic statistics
# ---------------------------------------------------------

print("\n========== STATISTICS ==========\n")

print("Characters :", len(text))
print("Words     :", len(text.split()))
print("Lines     :", len(text.splitlines()))





def parse_text_file(file_path: str) -> str:

    path = Path(file_path)

    return path.read_text(
        encoding="utf-8"
    )


document_path1 = (
    "phase-06-rag/module-01-document-processing/"
    "documents/spark_troubleshooting.txt"
)

text1 = parse_text_file(document_path1)

print(text1)