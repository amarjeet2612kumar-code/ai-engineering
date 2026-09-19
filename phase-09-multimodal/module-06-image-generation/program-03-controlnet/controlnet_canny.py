import cv2
from pathlib import Path


# ---------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------

INPUT_IMAGE = Path("data/input.jpg")

OUTPUT_DIR = Path("output")

EDGE_IMAGE = OUTPUT_DIR / "canny_edges.png"


# Create output directory if it doesn't exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. Read the original image
# ---------------------------------------------------------

image = cv2.imread(str(INPUT_IMAGE))

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {INPUT_IMAGE}"
    )


print("Original image loaded.")


# ---------------------------------------------------------
# 3. Convert image to grayscale
# ---------------------------------------------------------

# Canny works on a single-channel grayscale image.
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ---------------------------------------------------------
# 4. Detect edges using Canny
# ---------------------------------------------------------

edges = cv2.Canny(
    gray,
    threshold1=100,
    threshold2=200
)


# ---------------------------------------------------------
# 5. Save the edge map
# ---------------------------------------------------------

cv2.imwrite(
    str(EDGE_IMAGE),
    edges
)


print("Canny edge map created.")
print("Saved to:", EDGE_IMAGE)