# Import os for working with directories and file paths.
import os

# Import csv for reading CSV tables.
import csv

# Import NumPy for storing embeddings.
import numpy as np

# Import FAISS for vector similarity search.
import faiss

# Import SentenceTransformer for generating text embeddings.
from sentence_transformers import SentenceTransformer


# =========================================================
# 1. Configuration
# =========================================================

# Directory containing our CSV tables.
TABLE_DIR = "data/tables"

# Query entered by the user.
QUERY = "Which table contains product sales and amounts?"

# Number of tables to retrieve.
TOP_K = 3

# Lightweight sentence embedding model.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================
# 2. Load embedding model
# =========================================================

print("Loading embedding model...")

# Load the pretrained sentence embedding model.
model = SentenceTransformer(
    MODEL_NAME
)

print("Embedding model loaded.")


# =========================================================
# 3. Read tables
# =========================================================

print("\nReading tables...")


# List that will contain table metadata.
tables = []


# Find all CSV files.
for filename in sorted(os.listdir(TABLE_DIR)):

    # Ignore files that are not CSV files.
    if not filename.lower().endswith(".csv"):
        continue

    # Build the complete file path.
    table_path = os.path.join(
        TABLE_DIR,
        filename
    )

    # Open the CSV file.
    with open(
        table_path,
        "r",
        encoding="utf-8"
    ) as file:

        # Read CSV rows as dictionaries.
        reader = csv.DictReader(file)

        # Convert the reader into a list.
        rows = list(reader)

        # Get column names.
        columns = reader.fieldnames


    # Create a table record.
    table = {
        "table_id": os.path.splitext(filename)[0],
        "filename": filename,
        "path": table_path,
        "columns": columns,
        "rows": rows
    }

    # Add the table metadata to our list.
    tables.append(table)


# Display number of tables.
print(
    f"Found {len(tables)} tables."
)


# =========================================================
# 4. Convert table into text
# =========================================================

def table_to_text(table):
    """
    Convert a structured table into a text representation
    that can be converted into an embedding.
    """

    # Get the column names.
    columns = table["columns"]

    # Get the table rows.
    rows = table["rows"]

    # Start with the table name.
    text = (
        f"Table: {table['table_id']}\n"
    )

    # Add the column names.
    text += (
        "Columns: "
        + ", ".join(columns)
        + "\n"
    )

    # Add each row.
    for row in rows:

        # Store each column-value pair.
        row_parts = []

        # Process every column.
        for column in columns:

            # Get the value for this column.
            value = row[column]

            # Preserve the relationship:
            #
            # Product = Laptop
            #
            row_parts.append(
                f"{column} = {value}"
            )

        # Add the complete row.
        text += (
            "Row: "
            + ", ".join(row_parts)
            + "\n"
        )

    return text


# =========================================================
# 5. Create table text representations
# =========================================================

print(
    "\nCreating table representations..."
)


# Store text representations.
table_texts = []


# Convert every table to text.
for table in tables:

    # Convert table into structured text.
    text = table_to_text(
        table
    )

    # Store the representation.
    table_texts.append(
        text
    )

    # Display what we created.
    print(
        f"\n--- {table['table_id']} ---"
    )

    print(text)


# =========================================================
# 6. Generate table embeddings
# =========================================================

print(
    "\nGenerating table embeddings..."
)


# Convert all table representations into embeddings.
table_embeddings = model.encode(
    table_texts,
    normalize_embeddings=True
)


# Convert to float32 for FAISS.
table_embeddings = np.array(
    table_embeddings
).astype("float32")


# Display embedding shape.
print(
    "Table embedding matrix shape:",
    table_embeddings.shape
)


# =========================================================
# 7. Create FAISS index
# =========================================================

# Get embedding dimension.
embedding_dimension = table_embeddings.shape[1]


# Create FAISS Inner Product index.
#
# Because embeddings are normalized:
#
# Inner Product = Cosine Similarity
index = faiss.IndexFlatIP(
    embedding_dimension
)


# =========================================================
# 8. Store table embeddings in FAISS
# =========================================================

# Add all table embeddings to FAISS.
index.add(
    table_embeddings
)


# Display number of vectors stored.
print(
    "Tables stored in FAISS:",
    index.ntotal
)


# =========================================================
# 9. Generate query embedding
# =========================================================

print(
    f"\nQuery: '{QUERY}'"
)

print(
    "Generating query embedding..."
)


# Convert the query into an embedding.
query_embedding = model.encode(
    [QUERY],
    normalize_embeddings=True
)


# Convert to float32.
query_embedding = np.array(
    query_embedding
).astype("float32")


# =========================================================
# 10. Search FAISS
# =========================================================

print(
    "Searching table index..."
)


# Search for the most similar tables.
distances, indices = index.search(
    query_embedding,
    TOP_K
)


# =========================================================
# 11. Display retrieval results
# =========================================================

print(
    "\nTop matching tables:\n"
)


# Loop through retrieved tables.
for rank in range(TOP_K):

    # Get the table index.
    table_index = indices[0][rank]

    # Get similarity score.
    similarity_score = distances[0][rank]

    # Get the table metadata.
    table = tables[table_index]

    # Display result.
    print(
        f"{rank + 1}. "
        f"{table['filename']} "
        f"similarity: {similarity_score:.4f}"
    )


# =========================================================
# 12. Show the top retrieved table
# =========================================================

# Get the index of the best matching table.
best_table_index = indices[0][0]

# Get the table.
best_table = tables[
    best_table_index
]


print(
    "\n=============================="
)

print(
    "RETRIEVED TABLE"
)

print(
    "=============================="
)


# Display table metadata.
print(
    "Table ID:",
    best_table["table_id"]
)

print(
    "Source:",
    best_table["path"]
)

print(
    "Columns:",
    best_table["columns"]
)


# =========================================================
# 13. Display retrieved rows
# =========================================================

print(
    "\nRows:"
)


# Display every row from the retrieved table.
for row in best_table["rows"]:

    print(
        row
    )


# =========================================================
# 14. Finished
# =========================================================

print(
    "\nTable retrieval completed."
)