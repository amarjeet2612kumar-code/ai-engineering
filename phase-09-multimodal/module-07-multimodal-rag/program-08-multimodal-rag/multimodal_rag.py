import os
import json
import base64

import faiss
import numpy as np
import torch

from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel


# =========================================================
# 1. CONFIGURATION
# =========================================================

# Load the .env file.
load_dotenv()

# Create OpenAI client.
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# CLIP model used for multimodal embeddings.
MODEL_NAME = "openai/clip-vit-base-patch32"

# Data directories.
TEXT_DIR = "data/text"
IMAGE_DIR = "data/images"

# Output files.
INDEX_FILE = "output/multimodal.faiss"
METADATA_FILE = "output/metadata.json"

# Number of results to retrieve.
TOP_K = 3


# =========================================================
# 2. LOAD CLIP MODEL
# =========================================================

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

print("CLIP model loaded.")


# =========================================================
# 3. TEXT EMBEDDING FUNCTION
# =========================================================

def get_text_embedding(text):
    """
    Convert text into a normalized CLIP embedding.
    """

    # Prepare the text for CLIP.
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    )

    # We only need inference, so gradients are disabled.
    with torch.no_grad():

        # Run the CLIP text encoder.
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        # Get the pooled text representation.
        pooled_output = text_outputs.pooler_output

        # Convert it into CLIP's shared embedding space.
        text_embedding = model.text_projection(
            pooled_output
        )

    # Normalize the embedding.
    text_embedding = text_embedding / text_embedding.norm(
        dim=-1,
        keepdim=True
    )

    # Convert PyTorch tensor to NumPy float32.
    return text_embedding.cpu().numpy().astype("float32")


# =========================================================
# 4. IMAGE EMBEDDING FUNCTION
# =========================================================

def get_image_embedding(image_path):
    """
    Convert an image into a normalized CLIP embedding.
    """

    # Open the image.
    image = Image.open(image_path).convert("RGB")

    # Prepare image for CLIP.
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Disable gradients.
    with torch.no_grad():

        # Run CLIP vision encoder.
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        # Get pooled image representation.
        pooled_output = vision_outputs.pooler_output

        # Convert it into CLIP's shared embedding space.
        image_embedding = model.visual_projection(
            pooled_output
        )

    # Normalize the embedding.
    image_embedding = image_embedding / image_embedding.norm(
        dim=-1,
        keepdim=True
    )

    # Convert to NumPy float32.
    return image_embedding.cpu().numpy().astype("float32")


# =========================================================
# 5. BUILD MULTIMODAL KNOWLEDGE BASE
# =========================================================

print("\n==============================")
print("BUILDING MULTIMODAL KNOWLEDGE BASE")
print("==============================")

vectors = []
metadata = []


# Get all text documents.
text_files = sorted(
    file
    for file in os.listdir(TEXT_DIR)
    if file.endswith(".txt")
)


# Process every document.
for text_file in text_files:

    # Example:
    # page_01.txt -> page_01
    document_id = text_file.replace(
        ".txt",
        ""
    )

    # Full text file path.
    text_path = os.path.join(
        TEXT_DIR,
        text_file
    )

    # Corresponding image.
    image_path = os.path.join(
        IMAGE_DIR,
        document_id + ".jpg"
    )

    print(f"\nProcessing {document_id}...")


    # -----------------------------------------------------
    # TEXT
    # -----------------------------------------------------

    # Read document text.
    with open(
        text_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read().strip()

    print("Creating text embedding...")

    # Create text embedding.
    text_embedding = get_text_embedding(text)

    # Store vector.
    vectors.append(
        text_embedding[0]
    )

    # Store metadata.
    metadata.append({
        "document_id": document_id,
        "modality": "text",
        "source": text_path,
        "content": text
    })


    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    print("Creating image embedding...")

    # Create image embedding.
    image_embedding = get_image_embedding(
        image_path
    )

    # Store vector.
    vectors.append(
        image_embedding[0]
    )

    # Store metadata.
    metadata.append({
        "document_id": document_id,
        "modality": "image",
        "source": image_path,
        "content": ""
    })


# =========================================================
# 6. CREATE EMBEDDING MATRIX
# =========================================================

embedding_matrix = np.array(
    vectors,
    dtype="float32"
)

print("\nEmbedding matrix shape:")
print(embedding_matrix.shape)


# =========================================================
# 7. CREATE FAISS INDEX
# =========================================================

# CLIP embedding dimension.
embedding_dimension = embedding_matrix.shape[1]

# Inner-product index.
#
# Because the vectors are normalized,
# inner product = cosine similarity.
index = faiss.IndexFlatIP(
    embedding_dimension
)

# Add all text and image embeddings.
index.add(
    embedding_matrix
)

print(
    "Vectors stored in FAISS:",
    index.ntotal
)


# =========================================================
# 8. SAVE FAISS INDEX
# =========================================================

faiss.write_index(
    index,
    INDEX_FILE
)

print(
    "FAISS index saved to:",
    INDEX_FILE
)


# =========================================================
# 9. SAVE METADATA
# =========================================================

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
    "Metadata saved to:",
    METADATA_FILE
)


# =========================================================
# 10. USER QUESTION
# =========================================================

QUESTION = (
    "Which page contains information "
    "about a laptop used for data engineering?"
)

print("\n==============================")
print("USER QUESTION")
print("==============================")

print(QUESTION)


# =========================================================
# 11. CREATE QUERY EMBEDDING
# =========================================================

print("\nCreating query embedding...")

query_embedding = get_text_embedding(
    QUESTION
)


# =========================================================
# 12. RETRIEVE FROM FAISS
# =========================================================

print("Searching multimodal index...")

scores, indices = index.search(
    query_embedding,
    TOP_K
)


# =========================================================
# 13. LOAD RETRIEVED METADATA
# =========================================================

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)


print("\n==============================")
print("RETRIEVED EVIDENCE")
print("==============================")


retrieved_items = []


for rank, vector_index in enumerate(
    indices[0]
):

    item = metadata[vector_index]

    score = scores[0][rank]

    print(
        f"\n{rank + 1}. "
        f"{item['document_id']} "
        f"| {item['modality']} "
        f"| similarity: {score:.4f}"
    )

    print(
        "Source:",
        item["source"]
    )

    retrieved_items.append(item)


# =========================================================
# 14. BUILD TEXT CONTEXT
# =========================================================

text_context = ""

for item in retrieved_items:

    # We only add actual text content here.
    if item["modality"] == "text":

        text_context += (
            f"\nDocument: "
            f"{item['document_id']}\n"
        )

        text_context += (
            f"Source: "
            f"{item['source']}\n"
        )

        text_context += (
            f"Content:\n"
            f"{item['content']}\n"
        )


# =========================================================
# 15. PREPARE RETRIEVED IMAGES
# =========================================================

image_inputs = []


for item in retrieved_items:

    # Skip text results.
    if item["modality"] != "image":
        continue

    # Get image path.
    image_path = item["source"]

    # Read image bytes.
    with open(
        image_path,
        "rb"
    ) as file:

        image_bytes = file.read()

    # Convert image bytes to Base64.
    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    # Add image to the multimodal request.
    image_inputs.append({
        "type": "input_image",
        "image_url": (
            "data:image/jpeg;base64,"
            + image_base64
        )
    })


# =========================================================
# 16. CREATE FINAL RAG PROMPT
# =========================================================

prompt = f"""
Answer the user's question using only the retrieved evidence.

If the evidence is insufficient, say:
"The retrieved evidence is insufficient."

User question:
{QUESTION}

Retrieved text evidence:
{text_context}

Give a short, factual answer.
"""


# =========================================================
# 17. SEND RETRIEVED EVIDENCE TO OPENAI
# =========================================================

print("\nGenerating final answer...")

# Start with the text prompt.
content = [
    {
        "type": "input_text",
        "text": prompt
    }
]

# Add retrieved images.
content.extend(
    image_inputs
)


response = client.responses.create(
    model="gpt-5-mini",
    input=[
        {
            "role": "user",
            "content": content
        }
    ]
)


# =========================================================
# 18. DISPLAY FINAL ANSWER
# =========================================================

print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(
    response.output_text
)

print(
    "\nMultimodal RAG completed."
)