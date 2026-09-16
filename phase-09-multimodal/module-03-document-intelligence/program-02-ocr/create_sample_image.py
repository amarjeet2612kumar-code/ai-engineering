"""
Create a simple document image for OCR testing.

The image contains text that Tesseract OCR will read.
"""

from PIL import Image, ImageDraw, ImageFont


OUTPUT_FILE = "data/sample_document.png"


# Create a white image.
image = Image.new(
    "RGB",
    (1000, 700),
    "white",
)

# Create a drawing object so that we can write text.
draw = ImageDraw.Draw(image)


# Use a standard font available on most Linux systems.
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

font = ImageFont.truetype(
    FONT_PATH,
    32,
)


# Text that we want OCR to recognize.
lines = [
    "BANK STATEMENT",
    "",
    "Customer Name: Amar Kumar",
    "Account Number: 1234567890",
    "Transaction Date: 16-09-2026",
    "Transaction Amount: 50000",
    "Transaction Type: CREDIT",
]


# Starting position for the text.
x = 80
y = 80

# Draw each line onto the image.
for line in lines:

    draw.text(
        (x, y),
        line,
        fill="black",
        font=font,
    )

    y += 70


# Save the image.
image.save(OUTPUT_FILE)

print(f"Sample document image created: {OUTPUT_FILE}")