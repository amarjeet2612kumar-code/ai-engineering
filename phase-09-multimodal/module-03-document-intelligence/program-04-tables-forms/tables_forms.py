"""
Program 4: Tables + Forms Extraction

Goal:
    Demonstrate how OCR bounding boxes can be used
    to reconstruct a simple table.

Pipeline:

    Image
      ↓
    OCR
      ↓
    Text + Bounding Boxes
      ↓
    Group words into rows
      ↓
    Assign words to columns
      ↓
    Structured Table
      ↓
    JSON

This is a learning implementation.

Real production systems use more sophisticated
layout/table extraction algorithms or models.
"""

from collections import defaultdict
import json

from PIL import Image
import pytesseract
from pytesseract import Output


# Path to our invoice image.
IMAGE_PATH = "data/sample_invoice.png"


print("=" * 70)
print("TABLE EXTRACTION USING OCR + BOUNDING BOXES")
print("=" * 70)


# ---------------------------------------------------------
# Step 1: Load image
# ---------------------------------------------------------

print("\nLoading invoice...")

image = Image.open(IMAGE_PATH)
image = image.convert("RGB")

print("Invoice loaded.")
print(f"Image size: {image.size}")


# ---------------------------------------------------------
# Step 2: Run OCR
# ---------------------------------------------------------

print("\nRunning OCR...")

ocr_data = pytesseract.image_to_data(
    image,
    output_type=Output.DICT,
)

print("OCR completed.")


# ---------------------------------------------------------
# Step 3: Collect detected words
# ---------------------------------------------------------

words = []

number_of_items = len(ocr_data["text"])


for i in range(number_of_items):

    text = ocr_data["text"][i].strip()

    if not text:
        continue

    confidence = float(ocr_data["conf"][i])

    if confidence < 30:
        continue

    x = ocr_data["left"][i]
    y = ocr_data["top"][i]
    width = ocr_data["width"][i]
    height = ocr_data["height"][i]

    words.append(
        {
            "text": text,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }
    )


# ---------------------------------------------------------
# Step 4: Group words into rows
# ---------------------------------------------------------

"""
Words belonging to the same table row should have
similar Y coordinates.

Example:

Laptop     1       50000
  ↑        ↑          ↑
same Y    same Y    same Y

Therefore, we group words whose Y positions
are sufficiently close.

The threshold is intentionally simple for learning.
"""

ROW_THRESHOLD = 25

rows = []

for word in words:

    placed = False

    for row in rows:

        # Compare the word's Y position with
        # the average Y position of the existing row.
        average_y = sum(
            item["y"] for item in row
        ) / len(row)

        if abs(word["y"] - average_y) <= ROW_THRESHOLD:

            row.append(word)
            placed = True
            break

    if not placed:

        # Start a new row.
        rows.append([word])


# Sort rows from top to bottom.
rows.sort(
    key=lambda row: min(item["y"] for item in row)
)


# Sort words within each row from left to right.
for row in rows:

    row.sort(
        key=lambda item: item["x"]
    )


# ---------------------------------------------------------
# Step 5: Display detected rows
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("DETECTED ROWS")
print("-" * 70)

for row in rows:

    row_text = " | ".join(
        item["text"] for item in row
    )

    print(row_text)


# ---------------------------------------------------------
# Step 6: Identify the table
# ---------------------------------------------------------

"""
Our document contains:

Product | Quantity | Amount

followed by three data rows.

For this learning example, we locate the
header row and use it to determine the
table columns.
"""

header_index = None

for index, row in enumerate(rows):

    row_text = [
        item["text"].lower()
        for item in row
    ]

    if (
        "product" in row_text
        and "quantity" in row_text
        and "amount" in row_text
    ):
        header_index = index
        break


if header_index is None:

    raise RuntimeError(
        "Table header could not be detected."
    )


header_row = rows[header_index]


# ---------------------------------------------------------
# Step 7: Determine column positions
# ---------------------------------------------------------

"""
The header tells us where each column is located.

For example:

Product     → x ≈ 120
Quantity    → x ≈ 500
Amount      → x ≈ 750

Later values can be assigned to the
nearest column.
"""

column_positions = {}

for item in header_row:

    column_name = item["text"].lower()

    if column_name == "product":

        column_positions["Product"] = item["x"]

    elif column_name == "quantity":

        column_positions["Quantity"] = item["x"]

    elif column_name == "amount":

        column_positions["Amount"] = item["x"]


# ---------------------------------------------------------
# Step 8: Extract table rows
# ---------------------------------------------------------

table_data = []


for row in rows[header_index + 1:]:

    # Skip the Total row.
    row_text = " ".join(
        item["text"] for item in row
    )

    if "total" in row_text.lower():
        continue

    record = {
        "Product": "",
        "Quantity": "",
        "Amount": "",
    }


    # Assign each detected word to the nearest
    # table column based on its X coordinate.
    for item in row:

        x = item["x"]

        nearest_column = min(
            column_positions,
            key=lambda column:
            abs(x - column_positions[column])
        )

        # Add the text to the selected column.
        if record[nearest_column]:

            record[nearest_column] += " " + item["text"]

        else:

            record[nearest_column] = item["text"]


    table_data.append(record)


# ---------------------------------------------------------
# Step 9: Display structured data
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("STRUCTURED TABLE")
print("-" * 70)

print(
    json.dumps(
        table_data,
        indent=4,
    )
)


print("\n" + "=" * 70)
print("TABLE EXTRACTION COMPLETED")
print("=" * 70)