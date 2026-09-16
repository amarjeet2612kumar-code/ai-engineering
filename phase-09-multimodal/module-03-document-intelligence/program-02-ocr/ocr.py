"""
Program 2: Optical Character Recognition (OCR)

Goal:
    Read text from an image using Tesseract OCR.

Input:
    Image containing text.

Output:
    Machine-readable text extracted from the image.

Technology:
    Tesseract OCR
    pytesseract
    Pillow
"""

from PIL import Image
import pytesseract


# Path of the image that contains text.
IMAGE_PATH = "data/sample_document.png"


print("=" * 70)
print("OCR - OPTICAL CHARACTER RECOGNITION")
print("=" * 70)


# ---------------------------------------------------------
# Step 1: Load the image
# ---------------------------------------------------------

print("\nLoading image...")

image = Image.open(IMAGE_PATH)

# Convert the image to RGB format.
image = image.convert("RGB")

print("Image loaded successfully.")
print(f"Image size: {image.size}")


# ---------------------------------------------------------
# Step 2: Run OCR
# ---------------------------------------------------------

print("\nRunning OCR...")

text = pytesseract.image_to_string(image)


# ---------------------------------------------------------
# Step 3: Display extracted text
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("EXTRACTED TEXT")
print("-" * 70)

print(text)


print("=" * 70)
print("OCR COMPLETED")
print("=" * 70)