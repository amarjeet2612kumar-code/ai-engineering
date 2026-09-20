# Import os for working with directories and file paths.
import os

# Import NumPy for storing embeddings and working with FAISS.
import numpy as np

# Import PyTorch for model inference.
import torch

# Import FAISS for vector similarity search.
import faiss

# Import Image from Pillow to open images.
from PIL import Image

# Import CLIP model and processor.
from transformers import CLIPProcessor, CLIPModel


# =========================================================
# 1. Configuration
# =========================================================

# Directory containing our image dataset.
IMAGE_DIR = "data/images"

# Text query entered by the user.
QUERY_TEXT = "a dog sitting on grass"

# Number of images we want to retrieve.
TOP_K = 5

# CLIP model.
MODEL_NAME = "openai/clip-vit-base-patch32"


# =========================================================
# 2. Load CLIP
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

# Put the model into evaluation mode.
model.eval()

print("CLIP model loaded.")


# =========================================================
# 3. Function: Generate image embedding
# =========================================================

def get_image_embedding(image):
    """
    Convert an image into a normalized CLIP image embedding.
    """

    # Prepare the image for the CLIP vision encoder.
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # We are doing inference, not training.
    # Therefore gradients are not required.
    with torch.no_grad():

        # Run the CLIP vision encoder.
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        # Extract the pooled visual representation.
        pooled_output = vision_outputs.pooler_output

        # Project the visual representation into
        # CLIP's final embedding space.
        image_embedding = model.visual_projection(
            pooled_output
        )

        # Normalize the embedding.
        #
        # This makes the vector length equal to 1.
        #
        # Therefore:
        #
        # Inner Product = Cosine Similarity
        image_embedding = image_embedding / image_embedding.norm(
            dim=-1,
            keepdim=True
        )

    # Move the tensor to CPU.
    image_embedding = image_embedding.cpu()

    # Detach it from the computation graph.
    image_embedding = image_embedding.detach()

    # Convert PyTorch tensor to NumPy.
    image_embedding = image_embedding.numpy()

    # FAISS expects float32.
    image_embedding = image_embedding.astype(
        "float32"
    )

    return image_embedding


# =========================================================
# 4. Function: Generate text embedding
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

    # Disable gradients because we are doing inference.
    with torch.no_grad():

        # Run the CLIP text encoder.
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        # Get the pooled text representation.
        pooled_output = text_outputs.pooler_output

        # Project the text representation into
        # CLIP's final embedding space.
        text_embedding = model.text_projection(
            pooled_output
        )

        # Normalize the text embedding.
        text_embedding = text_embedding / text_embedding.norm(
            dim=-1,
            keepdim=True
        )

    # Move the tensor to CPU.
    text_embedding = text_embedding.cpu()

    # Detach from the computation graph.
    text_embedding = text_embedding.detach()

    # Convert to NumPy.
    text_embedding = text_embedding.numpy()

    # FAISS expects float32.
    text_embedding = text_embedding.astype(
        "float32"
    )

    return text_embedding


# =========================================================
# 5. Find dataset images
# =========================================================

print("\nSearching for dataset images...")

# List that will contain image paths.
image_paths = []


# Read every file from the image directory.
for filename in os.listdir(IMAGE_DIR):

    # Convert filename to lowercase.
    filename_lower = filename.lower()

    # Check whether it is an image.
    if filename_lower.endswith(
        (".jpg", ".jpeg", ".png")
    ):

        # Create complete image path.
        image_path = os.path.join(
            IMAGE_DIR,
            filename
        )

        # Add image path to the list.
        image_paths.append(
            image_path
        )


# Sort the paths so the mapping between
# FAISS index positions and filenames stays predictable.
image_paths.sort()


# Display number of images.
print(
    f"Found {len(image_paths)} images."
)


# =========================================================
# 6. Generate image embeddings
# =========================================================

print("\nGenerating image embeddings...")

# List to store all image embeddings.
image_embeddings = []


# Process every image.
for image_path in image_paths:

    # Get filename for progress display.
    filename = os.path.basename(
        image_path
    )

    print(
        f"Processing: {filename}"
    )

    # Open the image.
    image = Image.open(
        image_path
    ).convert("RGB")

    # Generate image embedding.
    embedding = get_image_embedding(
        image
    )

    # Embedding shape is:
    #
    # [1, 512]
    #
    # We only need the first row.
    image_embeddings.append(
        embedding[0]
    )


# =========================================================
# 7. Convert embeddings into a matrix
# =========================================================

# Convert the list into a NumPy array.
#
# For 18 images:
#
# [18, 512]
image_embeddings = np.array(
    image_embeddings
).astype("float32")


# Display embedding shape.
print(
    "\nEmbedding matrix shape:",
    image_embeddings.shape
)


# =========================================================
# 8. Create FAISS vector index
# =========================================================

# Get embedding dimension.
#
# CLIP ViT-B/32 produces 512-dimensional
# projected embeddings.
embedding_dimension = image_embeddings.shape[1]


# Create a FAISS Inner Product index.
#
# Since embeddings are normalized:
#
# Inner Product = Cosine Similarity
index = faiss.IndexFlatIP(
    embedding_dimension
)


# =========================================================
# 9. Store image embeddings in FAISS
# =========================================================

# Add all image embeddings to FAISS.
index.add(
    image_embeddings
)


# Display number of vectors stored.
print(
    "Images stored in FAISS:",
    index.ntotal
)


# =========================================================
# 10. Display the text query
# =========================================================

print(
    f"\nText query: '{QUERY_TEXT}'"
)


# =========================================================
# 11. Convert text query into embedding
# =========================================================

print(
    "Generating text embedding..."
)

# Convert the text query into a CLIP text embedding.
query_embedding = get_text_embedding(
    QUERY_TEXT
)


# query_embedding shape:
#
# [1, 512]


# =========================================================
# 12. Search FAISS
# =========================================================

print(
    "Searching for matching images..."
)


# Search the image vectors using
# the text embedding as the query.
distances, indices = index.search(
    query_embedding,
    TOP_K
)


# =========================================================
# 13. Display results
# =========================================================

print(
    "\nTop matching images:\n"
)


# Display each result.
for rank in range(TOP_K):

    # Get the position of the matching image.
    image_index = indices[0][rank]

    # Get the similarity score.
    similarity_score = distances[0][rank]

    # Get the corresponding image path.
    image_path = image_paths[
        image_index
    ]

    # Get only the filename.
    filename = os.path.basename(
        image_path
    )

    # Display the result.
    print(
        f"{rank + 1}. {filename} "
        f"similarity: {similarity_score:.4f}"
    )


# =========================================================
# 14. Finished
# =========================================================

print(
    "\nText-to-image retrieval completed."
)