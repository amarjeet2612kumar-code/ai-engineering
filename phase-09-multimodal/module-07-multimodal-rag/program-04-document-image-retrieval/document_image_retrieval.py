# Import os for working with directories and file paths.
import os

# Import NumPy for storing embeddings and working with FAISS.
import numpy as np

# Import PyTorch for model inference.
import torch

# Import FAISS for vector similarity search.
import faiss

# Import Image from Pillow for opening images.
from PIL import Image

# Import CLIP model and processor.
from transformers import CLIPProcessor, CLIPModel


# =========================================================
# 1. Configuration
# =========================================================

# Directory containing document text files.
TEXT_DIR = "data/text"

# Directory containing document images.
IMAGE_DIR = "data/images"

# Text query entered by the user.
QUERY_TEXT = "laptop computer for data engineering"

# Number of results to retrieve from each modality.
TOP_K = 3

# Pretrained CLIP model.
MODEL_NAME = "openai/clip-vit-base-patch32"


# =========================================================
# 2. Load CLIP
# =========================================================

print("Loading CLIP model...")

# Load the pretrained CLIP model.
model = CLIPModel.from_pretrained(
    MODEL_NAME
)

# Load the processor.
processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

# We are doing inference, not training.
model.eval()

print("CLIP model loaded.")


# =========================================================
# 3. Generate image embedding
# =========================================================

def get_image_embedding(image):
    """
    Convert an image into a normalized CLIP image embedding.
    """

    # Prepare the image for CLIP.
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Disable gradient calculation.
    with torch.no_grad():

        # Run the CLIP vision encoder.
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        # Get the pooled visual representation.
        pooled_output = vision_outputs.pooler_output

        # Project the visual representation into
        # CLIP's shared embedding space.
        image_embedding = model.visual_projection(
            pooled_output
        )

        # Normalize the vector.
        image_embedding = image_embedding / image_embedding.norm(
            dim=-1,
            keepdim=True
        )

    # Move tensor to CPU.
    image_embedding = image_embedding.cpu()

    # Detach the tensor.
    image_embedding = image_embedding.detach()

    # Convert tensor to NumPy.
    image_embedding = image_embedding.numpy()

    # FAISS expects float32.
    image_embedding = image_embedding.astype(
        "float32"
    )

    return image_embedding


# =========================================================
# 4. Generate text embedding
# =========================================================

def get_text_embedding(text):
    """
    Convert text into a normalized CLIP text embedding.
    """

    # Prepare the text for CLIP.
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    )

    # Disable gradients during inference.
    with torch.no_grad():

        # Run the CLIP text encoder.
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        # Get the pooled text representation.
        pooled_output = text_outputs.pooler_output

        # Project into CLIP's shared embedding space.
        text_embedding = model.text_projection(
            pooled_output
        )

        # Normalize the embedding.
        text_embedding = text_embedding / text_embedding.norm(
            dim=-1,
            keepdim=True
        )

    # Move tensor to CPU.
    text_embedding = text_embedding.cpu()

    # Detach from computation graph.
    text_embedding = text_embedding.detach()

    # Convert to NumPy.
    text_embedding = text_embedding.numpy()

    # FAISS expects float32.
    text_embedding = text_embedding.astype(
        "float32"
    )

    return text_embedding


# =========================================================
# 5. Read document metadata
# =========================================================

print("\nReading documents...")


# This list will contain our document records.
documents = []


# Find all text files.
for filename in sorted(os.listdir(TEXT_DIR)):

    # Ignore files that are not .txt files.
    if not filename.endswith(".txt"):
        continue

    # Create the text file path.
    text_path = os.path.join(
        TEXT_DIR,
        filename
    )

    # Extract page number from filename.
    #
    # Example:
    #
    # page_01.txt → page_01
    document_id = os.path.splitext(
        filename
    )[0]

    # Read the text file.
    with open(
        text_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    # Build the corresponding image filename.
    image_filename = document_id + ".jpg"

    # Build the image path.
    image_path = os.path.join(
        IMAGE_DIR,
        image_filename
    )

    # Create a document record.
    document = {
        "document_id": document_id,
        "text": text,
        "text_path": text_path,
        "image_path": image_path
    }

    # Add the record to our document list.
    documents.append(document)


# Display number of documents.
print(
    f"Found {len(documents)} documents."
)


# =========================================================
# 6. Create text embeddings
# =========================================================

print("\nGenerating text embeddings...")

# Store text embeddings.
text_embeddings = []


# Process each document.
for document in documents:

    # Get document text.
    text = document["text"]

    # Generate text embedding.
    embedding = get_text_embedding(
        text
    )

    # Add the first row.
    text_embeddings.append(
        embedding[0]
    )


# Convert list to NumPy matrix.
#
# Example:
#
# [3 documents, 512 dimensions]
text_embeddings = np.array(
    text_embeddings
).astype("float32")


print(
    "Text embedding matrix shape:",
    text_embeddings.shape
)


# =========================================================
# 7. Create text FAISS index
# =========================================================

# Get embedding dimension.
embedding_dimension = text_embeddings.shape[1]


# Create FAISS index for text embeddings.
text_index = faiss.IndexFlatIP(
    embedding_dimension
)


# Add text embeddings to FAISS.
text_index.add(
    text_embeddings
)


print(
    "Text vectors stored in FAISS:",
    text_index.ntotal
)


# =========================================================
# 8. Create image embeddings
# =========================================================

print("\nGenerating image embeddings...")

# Store image embeddings.
image_embeddings = []


# Process every document image.
for document in documents:

    # Get image path.
    image_path = document["image_path"]

    # Open image.
    image = Image.open(
        image_path
    ).convert("RGB")

    # Generate image embedding.
    embedding = get_image_embedding(
        image
    )

    # Store first row.
    image_embeddings.append(
        embedding[0]
    )


# Convert to NumPy matrix.
image_embeddings = np.array(
    image_embeddings
).astype("float32")


print(
    "Image embedding matrix shape:",
    image_embeddings.shape
)


# =========================================================
# 9. Create image FAISS index
# =========================================================

# Create a separate FAISS index for images.
image_index = faiss.IndexFlatIP(
    embedding_dimension
)


# Add image embeddings.
image_index.add(
    image_embeddings
)


print(
    "Image vectors stored in FAISS:",
    image_index.ntotal
)


# =========================================================
# 10. Convert user query into embedding
# =========================================================

print(
    f"\nQuery: '{QUERY_TEXT}'"
)


print(
    "Generating query embedding..."
)


# Convert text query into a CLIP text embedding.
query_embedding = get_text_embedding(
    QUERY_TEXT
)


# =========================================================
# 11. Search text index
# =========================================================

print(
    "\nSearching text evidence..."
)


# Search the text vectors.
text_scores, text_indices = text_index.search(
    query_embedding,
    TOP_K
)


# =========================================================
# 12. Search image index
# =========================================================

print(
    "Searching image evidence..."
)


# Search image vectors using the same
# text query embedding.
image_scores, image_indices = image_index.search(
    query_embedding,
    TOP_K
)


# =========================================================
# 13. Display text results
# =========================================================

print(
    "\n=============================="
)

print(
    "TEXT RESULTS"
)

print(
    "=============================="
)


for rank in range(TOP_K):

    # Get document position.
    index_position = text_indices[0][rank]

    # Get similarity score.
    score = text_scores[0][rank]

    # Get document metadata.
    document = documents[index_position]

    # Display result.
    print(
        f"{rank + 1}. "
        f"{document['document_id']} "
        f"similarity: {score:.4f}"
    )


# =========================================================
# 14. Display image results
# =========================================================

print(
    "\n=============================="
)

print(
    "IMAGE RESULTS"
)

print(
    "=============================="
)


for rank in range(TOP_K):

    # Get document position.
    index_position = image_indices[0][rank]

    # Get similarity score.
    score = image_scores[0][rank]

    # Get document metadata.
    document = documents[index_position]

    # Get image filename.
    image_filename = os.path.basename(
        document["image_path"]
    )

    # Display result.
    print(
        f"{rank + 1}. "
        f"{image_filename} "
        f"similarity: {score:.4f}"
    )


# =========================================================
# 15. Display metadata mapping
# =========================================================

print(
    "\n=============================="
)

print(
    "RETRIEVED METADATA"
)

print(
    "=============================="
)


# Display the top text result.
top_text_index = text_indices[0][0]

top_text_document = documents[
    top_text_index
]


print(
    "\nTop text result:"
)

print(
    "Document ID:",
    top_text_document["document_id"]
)

print(
    "Text source:",
    top_text_document["text_path"]
)


# Display the top image result.
top_image_index = image_indices[0][0]

top_image_document = documents[
    top_image_index
]


print(
    "\nTop image result:"
)

print(
    "Document ID:",
    top_image_document["document_id"]
)

print(
    "Image source:",
    top_image_document["image_path"]
)


# =========================================================
# 16. Finished
# =========================================================

print(
    "\nDocument + image retrieval completed."
)