# Import os for working with directories and file paths.
import os

# Import NumPy for storing embeddings and working with FAISS.
import numpy as np

# Import PyTorch for tensor operations and inference.
import torch

# Import FAISS for vector similarity search.
import faiss

# Import Image from Pillow to open image files.
from PIL import Image

# Import CLIP model and processor from Hugging Face.
from transformers import CLIPProcessor, CLIPModel


# =========================================================
# 1. Configuration
# =========================================================

# Directory containing the images that we want to search.
IMAGE_DIR = "data/images"

# Image that we will use as the search/query image.
QUERY_IMAGE = "data/query.jpg"

# Number of similar images to retrieve.
TOP_K = 5

# Pretrained CLIP model.
MODEL_NAME = "openai/clip-vit-base-patch32"


# =========================================================
# 2. Load CLIP model and processor
# =========================================================

print("Loading CLIP model...")

# Load the pretrained CLIP model.
model = CLIPModel.from_pretrained(MODEL_NAME)

# Load the processor used to prepare images.
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# Put the model into evaluation mode.
#
# We are doing inference, not training.
model.eval()

print("CLIP model loaded.")


# =========================================================
# 3. Function to create an image embedding
# =========================================================

def get_image_embedding(image):
    """
    Convert an image into a normalized CLIP embedding.
    """

    # Prepare the image for the CLIP vision encoder.
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # -----------------------------------------------------
    # Run the vision encoder.
    # -----------------------------------------------------
    #
    # We only provide image input here.
    #
    # We do NOT use:
    #
    #     model(**inputs)
    #
    # because the complete CLIP model expects both
    # image and text inputs in this Transformers version.
    #
    # Instead, we directly run the vision model.
    with torch.no_grad():

        # Run the CLIP vision encoder.
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        # Extract the pooled visual representation.
        #
        # Shape:
        #
        # [1, 768]
        pooled_output = vision_outputs.pooler_output

        # Apply CLIP's visual projection layer.
        #
        # This converts the visual representation
        # into CLIP's final embedding space.
        #
        # Shape:
        #
        # [1, 512]
        image_embedding = model.visual_projection(
            pooled_output
        )

        # Normalize the embedding.
        #
        # After normalization, the vector has length 1.
        #
        # This allows us to use dot product in FAISS
        # as cosine similarity.
        image_embedding = image_embedding / image_embedding.norm(
            dim=-1,
            keepdim=True
        )

    # -----------------------------------------------------
    # Convert PyTorch tensor to NumPy
    # -----------------------------------------------------

    # Move the tensor to CPU.
    image_embedding = image_embedding.cpu()

    # Detach the tensor from the computation graph.
    #
    # This is required before calling .numpy().
    image_embedding = image_embedding.detach()

    # Convert the tensor into a NumPy array.
    image_embedding = image_embedding.numpy()

    # FAISS expects float32 values.
    image_embedding = image_embedding.astype(
        "float32"
    )

    # Return the final embedding.
    return image_embedding


# =========================================================
# 4. Find all dataset images
# =========================================================

print("\nSearching for dataset images...")

# Create an empty list to store image paths.
image_paths = []


# Read all files inside the image directory.
for filename in os.listdir(IMAGE_DIR):

    # Convert filename to lowercase.
    filename_lower = filename.lower()

    # Check whether the file is an image.
    if filename_lower.endswith(
        (".jpg", ".jpeg", ".png")
    ):

        # Create the complete image path.
        image_path = os.path.join(
            IMAGE_DIR,
            filename
        )

        # Add the path to our list.
        image_paths.append(
            image_path
        )


# Sort the image paths.
#
# This gives us a predictable mapping between:
#
# FAISS index position
#         ↓
# image_paths position
#
image_paths.sort()


# Print number of images found.
print(
    f"Found {len(image_paths)} images."
)


# =========================================================
# 5. Generate embeddings for dataset images
# =========================================================

print("\nGenerating image embeddings...")

# List to store all image embeddings.
image_embeddings = []


# Process each image.
for image_path in image_paths:

    # Get only the filename.
    filename = os.path.basename(
        image_path
    )

    # Show progress.
    print(
        f"Processing: {filename}"
    )

    # Open the image.
    image = Image.open(
        image_path
    ).convert("RGB")

    # Generate the CLIP image embedding.
    embedding = get_image_embedding(
        image
    )

    # The embedding has shape:
    #
    # [1, 512]
    #
    # We only need the first row.
    image_embeddings.append(
        embedding[0]
    )


# =========================================================
# 6. Convert embeddings into NumPy matrix
# =========================================================

# Convert the list into a NumPy array.
#
# For our 18 images:
#
# [18, 512]
#
# For 1000 images:
#
# [1000, 512]
image_embeddings = np.array(
    image_embeddings
).astype("float32")


# Print embedding matrix shape.
print(
    "\nEmbedding matrix shape:",
    image_embeddings.shape
)


# =========================================================
# 7. Create FAISS index
# =========================================================

# Get the size of one embedding.
#
# For CLIP ViT-B/32:
#
# 512
embedding_dimension = image_embeddings.shape[1]


# Create a FAISS Inner Product index.
#
# Because our embeddings are normalized:
#
# Inner Product = Cosine Similarity
#
index = faiss.IndexFlatIP(
    embedding_dimension
)


# =========================================================
# 8. Add image embeddings to FAISS
# =========================================================

# Add all dataset embeddings to the FAISS index.
index.add(
    image_embeddings
)


# Print number of vectors stored.
print(
    "Images stored in FAISS:",
    index.ntotal
)


# =========================================================
# 9. Load query image
# =========================================================

print("\nLoading query image...")

# Open the query image.
query_image = Image.open(
    QUERY_IMAGE
).convert("RGB")


# =========================================================
# 10. Generate query embedding
# =========================================================

print("Generating query embedding...")

# Convert the query image into a CLIP embedding.
query_embedding = get_image_embedding(
    query_image
)


# query_embedding shape:
#
# [1, 512]


# =========================================================
# 11. Search FAISS
# =========================================================

print("Searching for similar images...")

# Search for the TOP_K most similar images.
#
# distances:
#     similarity scores
#
# indices:
#     positions of matching images
#
distances, indices = index.search(
    query_embedding,
    TOP_K
)


# =========================================================
# 12. Display search results
# =========================================================

print("\nTop similar images:\n")


# Loop through the retrieved results.
for rank in range(TOP_K):

    # Get the position of the matching image.
    image_index = indices[0][rank]

    # Get the similarity score.
    similarity_score = distances[0][rank]

    # Get the image path using the FAISS index.
    image_path = image_paths[
        image_index
    ]

    # Extract only the filename.
    filename = os.path.basename(
        image_path
    )

    # Display the result.
    print(
        f"{rank + 1}. {filename} "
        f"similarity: {similarity_score:.4f}"
    )


# =========================================================
# 13. Finished
# =========================================================

print(
    "\nImage retrieval completed."
)