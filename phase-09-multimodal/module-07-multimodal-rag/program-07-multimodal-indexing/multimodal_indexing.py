# Import os for working with files and directories.
import os

# Import json for saving and loading metadata.
import json

# Import NumPy for handling embeddings.
import numpy as np

# Import PyTorch for CLIP inference.
import torch

# Import FAISS for vector indexing and similarity search.
import faiss

# Import Image from Pillow for opening images.
from PIL import Image

# Import CLIP model and processor.
from transformers import CLIPProcessor, CLIPModel


# =========================================================
# 1. Configuration
# =========================================================

# Directory containing dataset images.
IMAGE_DIR = "data/images"

# Query image.
QUERY_IMAGE = "data/query.jpg"

# Directory where the index and metadata will be saved.
OUTPUT_DIR = "output"

# FAISS index file.
INDEX_FILE = "output/images.faiss"

# Metadata file.
METADATA_FILE = "output/image_metadata.json"

# Number of results to retrieve.
TOP_K = 5

# CLIP model.
MODEL_NAME = "openai/clip-vit-base-patch32"


# =========================================================
# 2. Load CLIP model
# =========================================================

print("Loading CLIP model...")

# Load the pretrained CLIP model.
model = CLIPModel.from_pretrained(
    MODEL_NAME
)

# Load the CLIP processor.
processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

# We are doing inference, not training.
model.eval()

print("CLIP model loaded.")


# =========================================================
# 3. Function to create image embedding
# =========================================================

def get_image_embedding(image):
    """
    Convert an image into a normalized CLIP embedding.
    """

    # Prepare the image for CLIP.
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Disable gradients because we are only doing inference.
    with torch.no_grad():

        # Run the CLIP vision encoder.
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        # Get the pooled visual representation.
        pooled_output = vision_outputs.pooler_output

        # Project it into CLIP's final embedding space.
        image_embedding = model.visual_projection(
            pooled_output
        )

        # Normalize the embedding.
        #
        # After normalization:
        # Inner Product = Cosine Similarity
        image_embedding = image_embedding / image_embedding.norm(
            dim=-1,
            keepdim=True
        )

    # Move the tensor to CPU.
    image_embedding = image_embedding.cpu()

    # Detach the tensor.
    image_embedding = image_embedding.detach()

    # Convert PyTorch tensor to NumPy.
    image_embedding = image_embedding.numpy()

    # FAISS requires float32.
    image_embedding = image_embedding.astype(
        "float32"
    )

    return image_embedding


# =========================================================
# 4. Create output directory
# =========================================================

# Create the output directory if it doesn't exist.
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# 5. INDEXING PHASE
# =========================================================

print("\n==============================")
print("INDEXING PHASE")
print("==============================")


# Store image paths.
image_paths = []


# Find all images in the dataset.
for filename in os.listdir(IMAGE_DIR):

    # Convert filename to lowercase.
    filename_lower = filename.lower()

    # Only process image files.
    if filename_lower.endswith(
        (".jpg", ".jpeg", ".png")
    ):

        # Build complete image path.
        image_path = os.path.join(
            IMAGE_DIR,
            filename
        )

        # Add path to the list.
        image_paths.append(
            image_path
        )


# Sort paths to keep a predictable mapping.
image_paths.sort()


# Display number of images.
print(
    f"Found {len(image_paths)} images."
)


# Store embeddings here.
image_embeddings = []


# Store metadata here.
metadata = []


# Process every image.
for image_path in image_paths:

    # Get filename.
    filename = os.path.basename(
        image_path
    )

    print(
        f"Creating embedding: {filename}"
    )

    # Open image.
    image = Image.open(
        image_path
    ).convert("RGB")

    # Generate embedding.
    embedding = get_image_embedding(
        image
    )

    # Add the embedding.
    #
    # Shape is [1, 512].
    #
    # We need the first row.
    image_embeddings.append(
        embedding[0]
    )

    # Create metadata for this vector.
    image_metadata = {
        "image_id": filename,
        "filename": filename,
        "source": image_path
    }

    # Store metadata.
    metadata.append(
        image_metadata
    )


# =========================================================
# 6. Convert embeddings to NumPy matrix
# =========================================================

# Convert list of embeddings into one matrix.
#
# For 18 images:
#
# [18, 512]
image_embeddings = np.array(
    image_embeddings
).astype("float32")


print(
    "\nEmbedding matrix shape:",
    image_embeddings.shape
)


# =========================================================
# 7. Create FAISS index
# =========================================================

# Get embedding dimension.
embedding_dimension = image_embeddings.shape[1]


# Create FAISS Inner Product index.
#
# Since vectors are normalized:
#
# Inner Product = Cosine Similarity
index = faiss.IndexFlatIP(
    embedding_dimension
)


# =========================================================
# 8. Add embeddings to FAISS
# =========================================================

# Add all image vectors.
index.add(
    image_embeddings
)


print(
    "Vectors stored in FAISS:",
    index.ntotal
)


# =========================================================
# 9. Save FAISS index
# =========================================================

# Save the FAISS index to disk.
faiss.write_index(
    index,
    INDEX_FILE
)


print(
    f"FAISS index saved to: {INDEX_FILE}"
)


# =========================================================
# 10. Save metadata
# =========================================================

# Save metadata as JSON.
with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=2
    )


print(
    f"Metadata saved to: {METADATA_FILE}"
)


# =========================================================
# 11. RETRIEVAL PHASE
# =========================================================

print("\n==============================")
print("RETRIEVAL PHASE")
print("==============================")


# ---------------------------------------------------------
# Load the FAISS index from disk.
# ---------------------------------------------------------

print(
    "Loading FAISS index..."
)


loaded_index = faiss.read_index(
    INDEX_FILE
)


print(
    f"Loaded vectors: {loaded_index.ntotal}"
)


# ---------------------------------------------------------
# Load metadata.
# ---------------------------------------------------------

print(
    "Loading metadata..."
)


with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    loaded_metadata = json.load(
        file
    )


print(
    f"Loaded metadata records: "
    f"{len(loaded_metadata)}"
)


# =========================================================
# 12. Load query image
# =========================================================

print(
    "\nLoading query image..."
)


# Open the query image.
query_image = Image.open(
    QUERY_IMAGE
).convert("RGB")


# =========================================================
# 13. Generate query embedding
# =========================================================

print(
    "Generating query embedding..."
)


# Convert query image into CLIP embedding.
query_embedding = get_image_embedding(
    query_image
)


# query_embedding shape:
#
# [1, 512]


# =========================================================
# 14. Search existing FAISS index
# =========================================================

print(
    "Searching existing FAISS index..."
)


# Search for the TOP_K nearest vectors.
distances, indices = loaded_index.search(
    query_embedding,
    TOP_K
)


# =========================================================
# 15. Display results
# =========================================================

print(
    "\nTop similar images:\n"
)


# Process each result.
for rank in range(TOP_K):

    # Get the vector position returned by FAISS.
    vector_index = indices[0][rank]

    # Get the similarity score.
    similarity_score = distances[0][rank]

    # Use the vector position to find metadata.
    result_metadata = loaded_metadata[
        vector_index
    ]

    # Display the result.
    print(
        f"{rank + 1}. "
        f"{result_metadata['filename']} "
        f"similarity: "
        f"{similarity_score:.4f}"
    )

    print(
        f"   Source: "
        f"{result_metadata['source']}"
    )


# =========================================================
# 16. Finished
# =========================================================

print(
    "\nMultimodal indexing and retrieval completed."
)