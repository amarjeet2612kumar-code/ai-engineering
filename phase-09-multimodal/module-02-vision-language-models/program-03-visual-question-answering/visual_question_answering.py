"""
Program 3: Visual Question Answering (VQA)

Goal:
    Give an image + a question to a Vision-Language Model (VLM)
    and generate an answer based on the visual content.

Input:
    Image + Question

Output:
    Natural-language answer

Model:
    HuggingFaceTB/SmolVLM-256M-Instruct

Example:

    Image:
        A dog sitting on grass

    Question:
        What animal is in the image?

    Answer:
        A dog.
"""


# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================

# PyTorch is used to run the neural network.
import torch

# PIL is used to load the image.
from PIL import Image

# AutoProcessor prepares the image and text for the model.
#
# AutoModelForImageTextToText loads a model capable of
# understanding images and generating text.
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)


# =========================================================
# 2. CONFIGURATION
# =========================================================

# We use the same small VLM from Program 2.
#
# Because the model is already downloaded, Hugging Face
# can reuse the cached model instead of downloading it again.
MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

# Path to our image.
IMAGE_PATH = "data/animal.jpg"

# Question that we want to ask about the image.
QUESTION = "What animal is in the image?"


# =========================================================
# 3. SELECT COMPUTING DEVICE
# =========================================================

# Check whether an NVIDIA GPU is available.
#
# If a CUDA-compatible GPU exists:
#     use GPU
#
# Otherwise:
#     use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("VISUAL QUESTION ANSWERING (VQA)")
print("=" * 70)

print(f"\nUsing device: {DEVICE}")


# =========================================================
# 4. LOAD IMAGE
# =========================================================

print("\nLoading image...")

# Open the local image file.
image = Image.open(IMAGE_PATH)

# Convert the image to RGB.
#
# This gives the model a standard 3-channel image.
image = image.convert("RGB")

print("Image loaded successfully.")
print(f"Image size: {image.size}")


# =========================================================
# 5. LOAD PROCESSOR
# =========================================================

print("\nLoading processor...")

# The processor prepares:
#
#     Image
#     +
#     Question
#
# into the format expected by the VLM.
processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

print("Processor loaded successfully.")


# =========================================================
# 6. LOAD VLM
# =========================================================

print("\nLoading VLM model...")

# Load the pre-trained Vision-Language Model.
#
# We are using the same model as Program 2.
# No training is being performed.
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME
)

# Move the model to CPU or GPU.
model = model.to(DEVICE)

# Put the model into evaluation mode.
#
# We are performing inference rather than training.
model.eval()

print("VLM model loaded successfully.")


# =========================================================
# 7. CREATE MULTIMODAL MESSAGE
# =========================================================

# The message contains two important pieces:
#
#     1. Image
#     2. Question
#
# The model receives both together.
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {
                "type": "text",
                "text": QUESTION,
            },
        ],
    }
]


# =========================================================
# 8. CREATE MODEL PROMPT
# =========================================================

print("\nPreparing question...")

# Convert the structured message into the model-specific
# chat format.
#
# add_generation_prompt=True tells the model that it should
# generate the assistant's answer.
prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

print(f"Question: {QUESTION}")


# =========================================================
# 9. PREPARE IMAGE + QUESTION
# =========================================================

# The processor converts the image and question into
# tensors that the neural network can process.
#
# return_tensors="pt"
#     means PyTorch tensors.
inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
)


# =========================================================
# 10. MOVE INPUTS TO MODEL DEVICE
# =========================================================

# The model and its input tensors must be on the same device.
#
# Example:
#
#     Model  → CPU
#     Inputs → CPU
#
# or:
#
#     Model  → GPU
#     Inputs → GPU
inputs = inputs.to(DEVICE)


# =========================================================
# 11. GENERATE ANSWER
# =========================================================

print("\nGenerating answer...")

# We only need inference.
#
# Therefore, gradients are disabled to reduce memory usage.
with torch.no_grad():

    # generate() produces the answer as token IDs.
    #
    # max_new_tokens controls how long the generated answer
    # is allowed to be.
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
    )


# =========================================================
# 12. DECODE TOKEN IDs
# =========================================================

# The model generates numerical token IDs.
#
# batch_decode() converts those token IDs into readable text.
generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]


# =========================================================
# 13. DISPLAY RESULT
# =========================================================

print("\n" + "=" * 70)
print("VQA RESULT")
print("=" * 70)

print(f"Question: {QUESTION}")
print(f"\nAnswer:\n{generated_text}")

print("=" * 70)