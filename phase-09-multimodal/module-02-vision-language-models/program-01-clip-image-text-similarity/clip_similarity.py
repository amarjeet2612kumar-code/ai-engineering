"""
Program 1: CLIP Image-Text Similarity

Goal:
    Understand how CLIP connects an image with text.

Input:
    1 image
    +
    multiple text descriptions

Output:
    Similarity score for each text description
    and the best matching description.

High-level flow:

    Image
      ↓
    CLIP Vision Encoder
      ↓
    Image Embedding
      ↓
    Compare with Text Embeddings
      ↑
    CLIP Text Encoder
      ↑
    Candidate Texts
"""


# ---------------------------------------------------------
# Import libraries
# ---------------------------------------------------------

# PyTorch is used to run the CLIP neural network.
import torch

# PIL is used to open the image file.
from PIL import Image

# AutoProcessor prepares the image and text for CLIP.
# CLIPModel loads the pre-trained CLIP model.
from transformers import AutoProcessor, CLIPModel


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Name of the pre-trained CLIP model available on Hugging Face.
MODEL_NAME = "openai/clip-vit-base-patch32"

# Path to the image that we want CLIP to understand.
IMAGE_PATH = "data/animal.jpg"

# These are the text descriptions that we want
# to compare against the image.
CANDIDATE_TEXTS = [
    "a dog sitting on grass",
    "a car on a road",
    "a mountain covered with snow",
    "a person riding a bicycle",
]


# ---------------------------------------------------------
# Load the CLIP model
# ---------------------------------------------------------

print("Loading CLIP model...")

# The processor converts raw images and text into tensors that the CLIP model understands.
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Load the pre-trained CLIP model.
#
# We are only using the model for inference.
# We are NOT training or fine-tuning it.
model = CLIPModel.from_pretrained(MODEL_NAME)

# Put the model into evaluation mode.
#
# This tells PyTorch that we are doing inference,
# not training.
model.eval()

print("CLIP model loaded successfully.")


# ---------------------------------------------------------
# Load the image
# ---------------------------------------------------------

print(f"\nLoading image: {IMAGE_PATH}")

# Open the image from the data directory.
image = Image.open(IMAGE_PATH)

# Convert the image to RGB.
#
# RGB means:
#     R = Red
#     G = Green
#     B = Blue
#
# This makes sure the image has the expected
# three color channels.
image = image.convert("RGB")

print(f"Image size: {image.size}")


# ---------------------------------------------------------
# Prepare image and text for CLIP
# ---------------------------------------------------------

print("\nPreparing image and text...")

# The processor prepares both:
#
#     image
#     candidate texts
#
# for the CLIP model.
#
# return_tensors="pt" means:
#     Return PyTorch tensors.
#
# padding=True means:
#     Make text inputs the same length where necessary.
inputs = processor(
    text=CANDIDATE_TEXTS,
    images=image,
    return_tensors="pt",
    padding=True,
)


# ---------------------------------------------------------
# Run CLIP inference
# ---------------------------------------------------------

print("Running CLIP inference...")

# We are only performing prediction.
#
# torch.no_grad() prevents PyTorch from storing gradients required for training.
#
# This reduces memory usage during inference.
with torch.no_grad():

    # Pass the prepared image and text into CLIP.
    #
    # CLIP processes:
    #
    #     image → image representation
    #
    #     text → text representations
    #
    # and then calculates image-text similarity.
    outputs = model(**inputs)


# ---------------------------------------------------------
# Get image and text embeddings
# ---------------------------------------------------------

# CLIP provides the final image embedding directly.
#
# Shape is approximately:
#
#     [number_of_images, embedding_dimension]
#
# We have one image, so there is only one image embedding.
image_embeddings = outputs.image_embeds

# CLIP also provides the final text embeddings.
#
# We have four candidate texts, so we get
# four text embeddings.
text_embeddings = outputs.text_embeds


# ---------------------------------------------------------
# Normalize embeddings
# ---------------------------------------------------------

# Normalize the image embedding.
#
# Normalization converts the vector into a unit vector.
# This allows us to compare the direction of the
# image and text embeddings.
image_embeddings = image_embeddings / image_embeddings.norm(
    dim=-1,
    keepdim=True,
)

# Normalize every text embedding in the same way.
text_embeddings = text_embeddings / text_embeddings.norm(
    dim=-1,
    keepdim=True,
)


# ---------------------------------------------------------
# Calculate image-text similarity
# ---------------------------------------------------------

# We calculate the dot product between:
#
#     image embedding
#             and
#     every text embedding
#
# Because the vectors are normalized, this dot product
# represents cosine similarity.
similarity_scores = image_embeddings @ text_embeddings.T

# We have only one image.
#
# Therefore, take the first row of the result.
similarity_scores = similarity_scores[0]


# ---------------------------------------------------------
# Display similarity scores
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("IMAGE-TEXT SIMILARITY RESULTS")
print("=" * 60)

# Convert PyTorch tensor values into normal Python numbers.
scores = similarity_scores.tolist()

# Display every candidate text together with its score.
for text, score in zip(CANDIDATE_TEXTS, scores):

    print(f"{text:<40} : {score:.4f}")


# ---------------------------------------------------------
# Find the best matching text
# ---------------------------------------------------------

# torch.argmax() returns the position of the
# highest similarity score.
best_index = torch.argmax(similarity_scores).item()

# Use that position to get the corresponding text.
best_match = CANDIDATE_TEXTS[best_index]

# Get the highest similarity score.
best_score = scores[best_index]


# ---------------------------------------------------------
# Display the final answer
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("BEST MATCH")
print("-" * 60)

print(f"Text  : {best_match}")
print(f"Score : {best_score:.4f}")

print("=" * 60)