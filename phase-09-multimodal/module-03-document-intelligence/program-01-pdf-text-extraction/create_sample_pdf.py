"""
Create a small text-based PDF for our PDF extraction program.

This file is only used to generate test data.

The actual PDF extraction will be implemented separately
in pdf_text_extraction.py.
"""

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# Output location of our sample PDF.
OUTPUT_FILE = "data/sample_invoice.pdf"


# Create a PDF canvas using A4 page size.
pdf = canvas.Canvas(OUTPUT_FILE, pagesize=A4)

# Get the width and height of the A4 page.
width, height = A4


# ---------------------------------------------------------
# Invoice Header
# ---------------------------------------------------------

pdf.setFont("Helvetica-Bold", 20)
pdf.drawString(50, height - 60, "Sample Invoice")


# ---------------------------------------------------------
# Customer Information
# ---------------------------------------------------------

pdf.setFont("Helvetica", 12)

pdf.drawString(50, height - 100, "Customer: Amar Kumar")
pdf.drawString(50, height - 120, "Invoice Number: INV-1001")
pdf.drawString(50, height - 140, "Invoice Date: 16-09-2026")


# ---------------------------------------------------------
# Product Information
# ---------------------------------------------------------

pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 190, "Product")
pdf.drawString(250, height - 190, "Quantity")
pdf.drawString(350, height - 190, "Amount")


pdf.setFont("Helvetica", 12)

pdf.drawString(50, height - 215, "Laptop")
pdf.drawString(250, height - 215, "1")
pdf.drawString(350, height - 215, "50000")


pdf.drawString(50, height - 240, "Monitor")
pdf.drawString(250, height - 240, "2")
pdf.drawString(350, height - 240, "30000")


# ---------------------------------------------------------
# Total
# ---------------------------------------------------------

pdf.setFont("Helvetica-Bold", 12)

pdf.drawString(50, height - 290, "Total Amount:")
pdf.drawString(350, height - 290, "80000")


# Finish and save the PDF.
pdf.save()

print(f"Sample PDF created successfully: {OUTPUT_FILE}")