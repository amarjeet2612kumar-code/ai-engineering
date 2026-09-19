import torch
from diffusers import StableDiffusionPipeline


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

# Pre-trained Stable Diffusion model.
# We are NOT training this model.
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# The text description of the image we want.
PROMPT = (
    "a futuristic data center in the mountains, "
    "modern architecture, realistic, daytime, "
    "high detail"
)

# Negative prompt tells the model what we want to avoid.
NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, ugly"
)

# Seed controls the random starting noise.
# Using the same seed helps reproduce the result.
SEED = 42

# Number of denoising steps.
# More steps generally means more computation.
NUM_STEPS = 20

# Image dimensions.
# We use a relatively small size because we are running locally.
WIDTH = 512
HEIGHT = 512


# ---------------------------------------------------------
# 2. Select device
# ---------------------------------------------------------

# Use GPU if CUDA is available.
# Otherwise use CPU.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)


# ---------------------------------------------------------
# 3. Load the pre-trained Stable Diffusion pipeline
# ---------------------------------------------------------

print("Loading Stable Diffusion model...")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

# Move the model to CPU/GPU.
pipe = pipe.to(device)

print("Model loaded.")


# ---------------------------------------------------------
# 4. Create a random generator using our seed
# ---------------------------------------------------------

generator = torch.Generator(device=device).manual_seed(SEED)


# ---------------------------------------------------------
# 5. Generate the image
# ---------------------------------------------------------

print("Generating image...")

result = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=NUM_STEPS,
    width=WIDTH,
    height=HEIGHT,
    generator=generator,
)

image = result.images[0]


# ---------------------------------------------------------
# 6. Save the generated image
# ---------------------------------------------------------

output_path = "output/generated_image.png"

image.save(output_path)

print("Image saved to:", output_path)