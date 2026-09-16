"""
Program 3: Document Layout Understanding

Goal:
    Detect text in a document and identify where each piece
    of text appears using bounding boxes.

Input:
    Image containing a document.

Output:
    Detected text + its position in the document.

Technology:
    Tesseract OCR
    pytesseract
    Pillow
"""

from PIL import Image
import pytesseract
from pytesseract import Output


# Path to our document image.
IMAGE_PATH = "data/sample_document.png"


print("=" * 70)
print("DOCUMENT LAYOUT UNDERSTANDING")
print("=" * 70)


# ---------------------------------------------------------
# Step 1: Load the document image
# ---------------------------------------------------------

print("\nLoading document...")

image = Image.open(IMAGE_PATH)
image = image.convert("RGB")

print("Document loaded successfully.")
print(f"Image size: {image.size}")


# ---------------------------------------------------------
# Step 2: Run OCR with layout information
# ---------------------------------------------------------

print("\nRunning OCR with bounding boxes...")

ocr_data = pytesseract.image_to_data(
    image,
    output_type=Output.DICT,
)

print("OCR completed.")


# ---------------------------------------------------------
# Step 3: Extract detected words and positions
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("DETECTED TEXT AND POSITIONS")
print("-" * 70)


# Tesseract returns multiple pieces of information.
# All of them have the same length.
number_of_items = len(ocr_data["text"])


for i in range(number_of_items):

    text = ocr_data["text"][i].strip()

    # Tesseract also provides a confidence score.
    confidence = float(ocr_data["conf"][i])

    # Ignore empty results.
    if not text:
        continue

    # Ignore extremely low-confidence detections.
    if confidence < 30:
        continue

    # Bounding box information.
    x = ocr_data["left"][i]
    y = ocr_data["top"][i]
    width = ocr_data["width"][i]
    height = ocr_data["height"][i]

    # Convert:
    #
    # (x, y, width, height)
    #
    # into:
    #
    # (x1, y1, x2, y2)

    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height

    print(f"\nText       : {text}")
    print(f"Confidence : {confidence:.2f}")
    print(f"Bounding Box: ({x1}, {y1}, {x2}, {y2})")


print("\n" + "=" * 70)
print("DOCUMENT LAYOUT EXTRACTION COMPLETED")
print("=" * 70)