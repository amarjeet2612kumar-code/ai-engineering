import torch

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

from PIL import Image
from pathlib import Path


# ---------------------------------------------------------
# 1. Model configuration
# ---------------------------------------------------------

# Stable Diffusion base model.
# This is the same model we used in Program 1.
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

# ControlNet specifically trained to understand Canny edges.
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"


# ---------------------------------------------------------
# 2. File paths
# ---------------------------------------------------------

CONTROL_IMAGE_PATH = Path("output/canny_edges.png")

OUTPUT_IMAGE_PATH = Path(
    "output/controlnet_generated.png"
)

OUTPUT_IMAGE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# 3. Text prompt
# ---------------------------------------------------------

PROMPT = (
    "a futuristic modern building in a mountain landscape, "
    "glass architecture, white and blue colors, "
    "photorealistic, daytime, highly detailed"
)

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed"
)


# ---------------------------------------------------------
# 4. Select device
# ---------------------------------------------------------

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)


# ---------------------------------------------------------
# 5. Load the ControlNet model
# ---------------------------------------------------------

print("Loading Canny ControlNet...")

# ControlNet is an additional neural network that understands
# the Canny edge image and converts it into conditioning
# information for the diffusion model.
#
# float32 is used on CPU because your laptop is CPU-based.
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float32,
)

print("ControlNet loaded.")


# ---------------------------------------------------------
# 6. Load Stable Diffusion + ControlNet
# ---------------------------------------------------------

print("Loading Stable Diffusion pipeline...")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float32,
)

# Move the pipeline to CPU.
pipe = pipe.to(device)


# ---------------------------------------------------------
# 7. Use a faster scheduler
# ---------------------------------------------------------

# The scheduler controls how the denoising process progresses.
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)


# ---------------------------------------------------------
# 8. Enable attention slicing
# ---------------------------------------------------------

# This reduces memory usage at the cost of some speed.
# It is useful on machines with limited RAM.
pipe.enable_attention_slicing()


# ---------------------------------------------------------
# 9. Load the Canny control image
# ---------------------------------------------------------

canny_image = Image.open(
    CONTROL_IMAGE_PATH
).convert("RGB")


# ---------------------------------------------------------
# 10. Resize the control image
# ---------------------------------------------------------

# We deliberately use 384x384 instead of 512x512.
# This reduces CPU memory usage and generation time.
canny_image = canny_image.resize(
    (384, 384)
)


# ---------------------------------------------------------
# 11. Generate the image
# ---------------------------------------------------------

print("Generating ControlNet image...")
print("This may take several minutes on CPU.")


generator = torch.Generator(
    device=device
).manual_seed(42)


result = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,

    # This is our structural control image.
    image=canny_image,

    # Number of diffusion/denoising steps.
    num_inference_steps=10,

    # Controls how strongly ControlNet influences
    # the generated image.
    controlnet_conditioning_scale=1.0,

    # Reproducible random starting point.
    generator=generator,

    # Output resolution.
    width=384,
    height=384,
)


# ---------------------------------------------------------
# 12. Save generated image
# ---------------------------------------------------------

generated_image = result.images[0]

generated_image.save(
    OUTPUT_IMAGE_PATH
)


print()
print("ControlNet generation completed.")
print("Saved to:", OUTPUT_IMAGE_PATH)