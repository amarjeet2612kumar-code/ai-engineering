# Import PyTorch for working with tensors and mathematical operations.
import torch

# Import PIL so that we can open the image file.
from PIL import Image

# Import the CLIP model and processor from Hugging Face Transformers.
from transformers import CLIPProcessor, CLIPModel


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

# Path to the image that we want to analyze.
IMAGE_PATH = "data/animal.jpg"

# List of text descriptions that we want to compare with the image.
TEXTS = [
    "a dog sitting on grass",
    "a cat sitting on grass",
    "a car on a road",
    "a mountain covered with snow",
]


# ---------------------------------------------------------
# 2. Load the CLIP model
# ---------------------------------------------------------

# This CLIP model contains:
# - an image encoder
# - a text encoder
#
# Both encoders produce embeddings that can be compared.
MODEL_NAME = "openai/clip-vit-base-patch32"

# Load the pretrained CLIP model.
model = CLIPModel.from_pretrained(MODEL_NAME)

# Load the processor used to prepare images and text
# in the format expected by CLIP.
processor = CLIPProcessor.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------
# 3. Load the image
# ---------------------------------------------------------

# Open the image from the data directory.
image = Image.open(IMAGE_PATH).convert("RGB")


# ---------------------------------------------------------
# 4. Prepare image and text for CLIP
# ---------------------------------------------------------

# The processor converts:
# - the image into image tensors
# - the text into token tensors
#
# return_tensors="pt" means that PyTorch tensors will be returned.
inputs = processor(
    text=TEXTS,
    images=image,
    return_tensors="pt",
    padding=True,
)


# ---------------------------------------------------------
# 5. Run the CLIP model
# ---------------------------------------------------------

# Pass the prepared inputs to the model.
#
# The model processes both:
# - the image
# - the text descriptions
#
# and creates embeddings for them.
outputs = model(**inputs)


# ---------------------------------------------------------
# 6. Extract image and text embeddings
# ---------------------------------------------------------

# Extract the image embedding from the model output.
image_embeddings = outputs.image_embeds

# Extract the text embeddings from the model output.
text_embeddings = outputs.text_embeds


# ---------------------------------------------------------
# 7. Display embedding dimensions
# ---------------------------------------------------------

# Print the shape of the image embedding.
print("Image embedding shape:", image_embeddings.shape)

# Print the shape of the text embeddings.
print("Text embedding shape:", text_embeddings.shape)


# ---------------------------------------------------------
# 8. Normalize the embeddings
# ---------------------------------------------------------

# Normalize the image embedding.
#
# dim=-1 means we normalize along the embedding dimension.
# keepdim=True keeps the tensor shape unchanged.
image_embeddings = image_embeddings / image_embeddings.norm(
    dim=-1,
    keepdim=True
)

# Normalize all text embeddings.
text_embeddings = text_embeddings / text_embeddings.norm(
    dim=-1,
    keepdim=True
)


# ---------------------------------------------------------
# 9. Calculate similarity
# ---------------------------------------------------------

# Calculate the similarity between the image embedding
# and every text embedding.
#
# Because the vectors are normalized, the dot product
# gives us cosine similarity.
similarities = image_embeddings @ text_embeddings.T


# ---------------------------------------------------------
# 10. Print the similarity scores
# ---------------------------------------------------------

print("\nImage-to-text similarity:\n")

# Convert the similarity tensor into a simple list of values.
scores = similarities[0].tolist()

# Print every text description with its similarity score.
for text, score in zip(TEXTS, scores):
    print(f"{text}: {score:.4f}")


# ---------------------------------------------------------
# 11. Find the best matching text
# ---------------------------------------------------------

# Find the position of the highest similarity score.
best_index = similarities[0].argmax().item()

# Get the corresponding text description.
best_text = TEXTS[best_index]

# Get the corresponding similarity score.
best_score = scores[best_index]


# Print the best match.
print("\nBEST MATCH:")
print(f"{best_text}: {best_score:.4f}")