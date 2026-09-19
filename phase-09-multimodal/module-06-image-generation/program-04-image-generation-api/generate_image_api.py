import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------

# Reads OPENAI_API_KEY from the .env file.
load_dotenv()


# ---------------------------------------------------------
# 2. Create OpenAI client
# ---------------------------------------------------------

# The SDK automatically uses OPENAI_API_KEY.
client = OpenAI()


# ---------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------

PROMPT = (
    "A futuristic data center built in the mountains, "
    "surrounded by green forests, modern white architecture, "
    "realistic photography, daytime, highly detailed"
)

OUTPUT_DIR = Path("output")

OUTPUT_FILE = OUTPUT_DIR / "generated_api_image.png"


# Create output directory if it doesn't exist.
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# 4. Send image generation request
# ---------------------------------------------------------

print("Sending request to image generation API...")

result = client.images.generate(
    model="gpt-image-2",
    prompt=PROMPT,
)


# ---------------------------------------------------------
# 5. Get Base64 image data
# ---------------------------------------------------------

# The API response contains the generated image
# as Base64-encoded data.
image_base64 = result.data[0].b64_json


# ---------------------------------------------------------
# 6. Convert Base64 back to binary image data
# ---------------------------------------------------------

image_bytes = base64.b64decode(
    image_base64
)


# ---------------------------------------------------------
# 7. Save image
# ---------------------------------------------------------

with open(OUTPUT_FILE, "wb") as file:
    file.write(image_bytes)


print("Image generation completed.")
print("Image saved to:", OUTPUT_FILE)