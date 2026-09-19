import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------
# 1. Load API key from the root .env file
# ---------------------------------------------------------

load_dotenv()

client = OpenAI()


# ---------------------------------------------------------
# 2. File paths
# ---------------------------------------------------------

INPUT_IMAGE = Path("data/input.jpg")
OUTPUT_IMAGE = Path("output/transformed_image.png")

# Create output directory if it doesn't exist.
OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 3. Transformation instruction
# ---------------------------------------------------------

PROMPT = """
Transform this image into a beautiful watercolor painting.

Keep the main objects, composition, and overall structure
of the original image recognizable.

Use soft watercolor textures, natural colors, and
hand-painted artistic details.
"""


# ---------------------------------------------------------
# 4. Open the original image
# ---------------------------------------------------------

with open(INPUT_IMAGE, "rb") as image_file:

    # Send the existing image together with our instruction.
    result = client.images.edit(
        model="gpt-image-2",
        image=image_file,
        prompt=PROMPT,
    )


# ---------------------------------------------------------
# 5. Extract the generated image
# ---------------------------------------------------------

image_base64 = result.data[0].b64_json

# Convert Base64 text back into image bytes.
image_bytes = base64.b64decode(image_base64)


# ---------------------------------------------------------
# 6. Save the generated image
# ---------------------------------------------------------

with open(OUTPUT_IMAGE, "wb") as output_file:
    output_file.write(image_bytes)


print("Image transformation completed.")
print("Output:", OUTPUT_IMAGE)