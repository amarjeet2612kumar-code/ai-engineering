"""
Program 4: Multimodal Reasoning using a Vision-Language Model

Goal:
    Give an image + a reasoning question to a VLM.

    The model should:
        1. Understand the image.
        2. Identify relevant visual evidence.
        3. Use that evidence to reason.
        4. Produce a conclusion with an explanation.

Input:
    Image + Reasoning Question

Output:
    Reasoned natural-language answer

Model:
    HuggingFaceTB/SmolVLM-256M-Instruct
"""


# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================

# PyTorch is used to run the neural network model.
import torch

# PIL is used to open and process our local image.
from PIL import Image

# AutoProcessor prepares the image and text for the VLM.
#
# AutoModelForImageTextToText loads a model that can
# accept image + text and generate text.
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)


# =========================================================
# 2. CONFIGURATION
# =========================================================

# We are using the same small VLM that we used in
# Program 2 and Program 3.
#
# Because the model is already cached on the machine,
# we should not need to download it again.
MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"


# Local image that we want the model to reason about.
IMAGE_PATH = "data/animal.jpg"


# This is a reasoning-oriented question.
#
# Notice that we are NOT simply asking:
#
#     "What animal is this?"
#
# Instead, we ask the model to:
#
#     1. Make a conclusion.
#     2. Explain the visual evidence.
#
# This is what makes this example different from
# our VQA program.
QUESTION = (
    "Look carefully at the image. "
    "Determine whether the dog is indoors or outdoors. "
    "Answer with either 'Indoors' or 'Outdoors', "
    "and then give one or two visual reasons for your answer."
)


# =========================================================
# 3. SELECT COMPUTING DEVICE
# =========================================================

# Check whether CUDA is available.
#
# CUDA means we have access to an NVIDIA GPU that
# PyTorch can use.
#
# If CUDA is not available, use CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("=" * 70)
print("MULTIMODAL REASONING")
print("=" * 70)

print(f"\nUsing device: {DEVICE}")


# =========================================================
# 4. LOAD IMAGE
# =========================================================

print("\nLoading image...")


# Open the image from our local data directory.
image = Image.open(IMAGE_PATH)


# Convert the image to RGB format.
#
# This ensures that the image has the standard
# Red + Green + Blue channels expected by the
# image-processing pipeline.
image = image.convert("RGB")


print("Image loaded successfully.")
print(f"Image size: {image.size}")


# =========================================================
# 5. LOAD PROCESSOR
# =========================================================

print("\nLoading processor...")


# The processor converts our raw inputs:
#
#     Image
#     +
#     Text
#
# into numerical representations that the model
# can process.
processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)


print("Processor loaded successfully.")


# =========================================================
# 6. LOAD VISION-LANGUAGE MODEL
# =========================================================

print("\nLoading VLM model...")


# Load the pre-trained SmolVLM model.
#
# Important:
#
# We are NOT training the model.
#
# We are using an already-trained model to perform
# inference.
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME
)


# Move the model to the selected device.
#
# Both model and input tensors must eventually be
# on the same device.
model = model.to(DEVICE)


# Put the model into evaluation mode.
#
# This tells PyTorch that we are performing inference,
# not training.
model.eval()


print("VLM model loaded successfully.")


# =========================================================
# 7. CREATE MULTIMODAL MESSAGE
# =========================================================

# Our message contains:
#
#     Image
#       +
#     Reasoning Question
#
# The image is represented by:
#
#     {"type": "image"}
#
# and the reasoning instruction is represented by:
#
#     {"type": "text", ...}
#
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

print("\nPreparing multimodal reasoning prompt...")


# apply_chat_template() converts our structured
# conversation into the format expected by SmolVLM.
#
# add_generation_prompt=True tells the model that
# it should now generate the assistant's response.
prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
)


print(f"Question: {QUESTION}")


# =========================================================
# 9. PREPARE IMAGE + TEXT
# =========================================================

# The processor now combines:
#
#     prompt
#       +
#     image
#
# and converts them into PyTorch tensors.
#
# return_tensors="pt"
# means:
#
#     "Return the result as PyTorch tensors."
inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
)


# =========================================================
# 10. MOVE INPUTS TO THE MODEL DEVICE
# =========================================================

# The model is on DEVICE.
#
# Therefore, the input tensors must also be on DEVICE.
#
# In your current environment this will most likely mean:
#
#     CPU → CPU
#
# If you later have a supported NVIDIA GPU:
#
#     GPU → GPU
inputs = inputs.to(DEVICE)


# =========================================================
# 11. GENERATE REASONED RESPONSE
# =========================================================

print("\nGenerating reasoning response...")
print("Please wait...")


# We are performing inference only.
#
# We don't need gradients because we are not training
# the model.
#
# Disabling gradients reduces unnecessary memory usage.
with torch.no_grad():

    # Ask the VLM to generate a response.
    #
    # max_new_tokens controls the maximum number of
    # new tokens that the model can generate.
    #
    # We use a relatively small value because your
    # laptop has limited resources.
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
    )


# =========================================================
# 12. DECODE THE GENERATED TOKENS
# =========================================================

# The neural network generates token IDs rather than
# directly returning a Python string.
#
# For example, conceptually:
#
#     [123, 456, 789, ...]
#
# batch_decode() converts those token IDs back into
# human-readable text.
generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]


# =========================================================
# 13. DISPLAY RESULT
# =========================================================

print("\n" + "=" * 70)
print("MULTIMODAL REASONING RESULT")
print("=" * 70)

print(f"Question:")
print(QUESTION)

print("\nModel Response:")
print(generated_text)

print("=" * 70)