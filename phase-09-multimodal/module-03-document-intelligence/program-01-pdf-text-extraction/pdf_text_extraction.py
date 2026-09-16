"""
Program 1: PDF Text Extraction

Goal:
    Read a text-based PDF and extract the text from each page.

Input:
    data/sample_invoice.pdf

Output:
    Extracted text printed to the terminal.

Library:
    pypdf
"""

from pypdf import PdfReader


# Path of the PDF that we want to read.
PDF_PATH = "data/sample_invoice.pdf"


print("=" * 70)
print("PDF TEXT EXTRACTION")
print("=" * 70)


# ---------------------------------------------------------
# Step 1: Load the PDF
# ---------------------------------------------------------

print("\nLoading PDF...")

reader = PdfReader(PDF_PATH)

print("PDF loaded successfully.")


# ---------------------------------------------------------
# Step 2: Find number of pages
# ---------------------------------------------------------

number_of_pages = len(reader.pages)

print(f"Number of pages: {number_of_pages}")


# ---------------------------------------------------------
# Step 3: Process each page
# ---------------------------------------------------------

print("\nExtracting text from PDF...")

for page_number, page in enumerate(reader.pages, start=1):

    print("\n" + "-" * 70)
    print(f"PAGE {page_number}")
    print("-" * 70)

    # Extract text from the current page.
    text = page.extract_text()

    # Print the extracted text.
    if text:
        print(text)
    else:
        print("No text could be extracted from this page.")


print("\n" + "=" * 70)
print("PDF EXTRACTION COMPLETED")
print("=" * 70)