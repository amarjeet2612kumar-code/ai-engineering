# Import os for working with directories and file paths.
import os

# Import json so that we can save chunk metadata.
import json

# Import NumPy for working with embeddings.
import numpy as np

# Import FAISS for vector indexing.
import faiss

# Import SentenceTransformer for text embeddings.
from sentence_transformers import SentenceTransformer


# =========================================================
# 1. Configuration
# =========================================================

# Directory containing document text.
TEXT_DIR = "data/text"

# Directory containing document images.
IMAGE_DIR = "data/images"

# Output directory.
OUTPUT_DIR = "output"

# Lightweight text embedding model.
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
# 3. Create multimodal chunks
# =========================================================

print("\nCreating multimodal chunks...")


# This list will contain all our chunks.
chunks = []


# Start chunk numbering from 1.
chunk_number = 1


# Find all text files.
for filename in sorted(os.listdir(TEXT_DIR)):

    # Ignore files that are not text files.
    if not filename.lower().endswith(".txt"):
        continue

    # Create the text file path.
    text_path = os.path.join(
        TEXT_DIR,
        filename
    )

    # Extract document ID.
    #
    # Example:
    #
    # page_01.txt → page_01
    document_id = os.path.splitext(
        filename
    )[0]

    # -----------------------------------------------------
    # Read the text
    # -----------------------------------------------------

    with open(
        text_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read().strip()


    # -----------------------------------------------------
    # Create text chunk
    # -----------------------------------------------------

    text_chunk = {
        "chunk_id": f"chunk_{chunk_number:03d}",
        "document_id": document_id,
        "parent_id": document_id,
        "page": document_id.replace(
            "page_",
            ""
        ),
        "modality": "text",
        "source": text_path,
        "content": text
    }


    # Add text chunk to our list.
    chunks.append(
        text_chunk
    )


    # Move to next chunk number.
    chunk_number += 1


    # -----------------------------------------------------
    # Create corresponding image chunk
    # -----------------------------------------------------

    image_filename = document_id + ".jpg"

    image_path = os.path.join(
        IMAGE_DIR,
        image_filename
    )


    # Check whether the corresponding image exists.
    if os.path.exists(image_path):

        image_chunk = {
            "chunk_id": f"chunk_{chunk_number:03d}",
            "document_id": document_id,
            "parent_id": document_id,
            "page": document_id.replace(
                "page_",
                ""
            ),
            "modality": "image",
            "source": image_path,
            "content": (
                f"Image associated with "
                f"{document_id}"
            )
        }


        # Add image chunk.
        chunks.append(
            image_chunk
        )


        # Move to next chunk number.
        chunk_number += 1


# =========================================================
# 4. Display created chunks
# =========================================================

print(
    f"Created {len(chunks)} chunks."
)


print(
    "\nChunk metadata:\n"
)


# Display every chunk.
for chunk in chunks:

    print(
        f"Chunk ID   : {chunk['chunk_id']}"
    )

    print(
        f"Document   : {chunk['document_id']}"
    )

    print(
        f"Parent     : {chunk['parent_id']}"
    )

    print(
        f"Page       : {chunk['page']}"
    )

    print(
        f"Modality   : {chunk['modality']}"
    )

    print(
        f"Source     : {chunk['source']}"
    )

    print(
        f"Content    : {chunk['content'][:100]}"
    )

    print(
        "-" * 50
    )


# =========================================================
# 5. Prepare content for embedding
# =========================================================

print(
    "\nPreparing chunks for embedding..."
)


# We can directly embed text chunks.

# For image chunks, we create a simple textual
# representation for this learning exercise.
#
# A production multimodal system would normally use
# an image encoder such as CLIP for actual image
# embeddings.
embedding_texts = []


for chunk in chunks:

    # If this is a text chunk, embed the actual text.
    if chunk["modality"] == "text":

        embedding_texts.append(
            chunk["content"]
        )

    # If this is an image chunk, embed its textual
    # metadata representation for now.
    else:

        embedding_texts.append(
            f"Image associated with "
            f"{chunk['document_id']}"
        )


# =========================================================
# 6. Generate embeddings
# =========================================================

print(
    "Generating embeddings..."
)


# Generate embeddings for all chunks.
embeddings = model.encode(
    embedding_texts,
    normalize_embeddings=True
)


# Convert to NumPy float32.
embeddings = np.array(
    embeddings
).astype("float32")


# Display shape.
print(
    "Embedding matrix shape:",
    embeddings.shape
)


# =========================================================
# 7. Create FAISS index
# =========================================================

# Get embedding dimension.
embedding_dimension = embeddings.shape[1]


# Create FAISS Inner Product index.
#
# Since embeddings are normalized:
#
# Inner Product = Cosine Similarity
index = faiss.IndexFlatIP(
    embedding_dimension
)


# =========================================================
# 8. Add chunk embeddings to FAISS
# =========================================================

# Add all chunk vectors.
index.add(
    embeddings
)


print(
    "Chunks stored in FAISS:",
    index.ntotal
)


# =========================================================
# 9. Save FAISS index
# =========================================================

# Create output directory if it does not exist.
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# Path where the FAISS index will be stored.
index_path = os.path.join(
    OUTPUT_DIR,
    "chunks.faiss"
)


# Save the FAISS index.
faiss.write_index(
    index,
    index_path
)


print(
    f"FAISS index saved to: {index_path}"
)


# =========================================================
# 10. Save metadata
# =========================================================

# Metadata file path.
metadata_path = os.path.join(
    OUTPUT_DIR,
    "chunk_metadata.json"
)


# Save chunk metadata as JSON.
with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        chunks,
        file,
        indent=2
    )


print(
    f"Metadata saved to: {metadata_path}"
)


# =========================================================
# 11. Display final architecture
# =========================================================

print(
    "\nMultimodal chunking completed."
)

print(
    "\nCreated:"
)

print(
    f"- {index_path}"
)

print(
    f"- {metadata_path}"
)