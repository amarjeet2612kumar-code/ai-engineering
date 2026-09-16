"""
Create a simple invoice image containing a table.

This image will be used to demonstrate how OCR
and bounding boxes can be used to reconstruct
structured table data.
"""

from PIL import Image, ImageDraw, ImageFont


OUTPUT_FILE = "data/sample_invoice.png"


# ---------------------------------------------------------
# Create image
# ---------------------------------------------------------

image = Image.new(
    "RGB",
    (1200, 800),
    "white",
)

draw = ImageDraw.Draw(image)


# Use a standard Linux font.
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

font = ImageFont.truetype(FONT_PATH, 32)
bold_font = ImageFont.truetype(FONT_PATH, 36)


# ---------------------------------------------------------
# Invoice title
# ---------------------------------------------------------

draw.text(
    (450, 50),
    "INVOICE",
    fill="black",
    font=bold_font,
)


# ---------------------------------------------------------
# Table coordinates
# ---------------------------------------------------------

# Column X positions.
product_x = 120
quantity_x = 500
amount_x = 750

# Row Y positions.
header_y = 150
row1_y = 230
row2_y = 310
row3_y = 390
total_y = 500


# ---------------------------------------------------------
# Table header
# ---------------------------------------------------------

draw.text(
    (product_x, header_y),
    "Product",
    fill="black",
    font=font,
)

draw.text(
    (quantity_x, header_y),
    "Quantity",
    fill="black",
    font=font,
)

draw.text(
    (amount_x, header_y),
    "Amount",
    fill="black",
    font=font,
)


# ---------------------------------------------------------
# Table rows
# ---------------------------------------------------------

rows = [
    ("Laptop", "1", "50000"),
    ("Monitor", "2", "30000"),
    ("Keyboard", "1", "2000"),
]

row_positions = [
    row1_y,
    row2_y,
    row3_y,
]


for row, y in zip(rows, row_positions):

    product, quantity, amount = row

    draw.text(
        (product_x, y),
        product,
        fill="black",
        font=font,
    )

    draw.text(
        (quantity_x, y),
        quantity,
        fill="black",
        font=font,
    )

    draw.text(
        (amount_x, y),
        amount,
        fill="black",
        font=font,
    )


# ---------------------------------------------------------
# Total
# ---------------------------------------------------------

draw.text(
    (750, total_y),
    "Total: 82000",
    fill="black",
    font=font,
)


# Save image.
image.save(OUTPUT_FILE)

print(f"Sample invoice created: {OUTPUT_FILE}")