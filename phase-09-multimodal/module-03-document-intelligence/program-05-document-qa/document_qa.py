"""
Program 5: Document Q&A

Goal:
    Extract text from a PDF and use an LLM to answer
    questions based only on the extracted document content.

Pipeline:

    PDF
      ↓
    pypdf
      ↓
    Extracted Text
      ↓
    Context
      ↓
    Question + Context
      ↓
    OpenAI API
      ↓
    Answer
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Path to the PDF.
PDF_PATH = "data/sample_invoice.pdf"

# Locate the .env file at the root of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

# Load environment variables from .env.
load_dotenv(ENV_PATH)


# ---------------------------------------------------------
# Check API key
# ---------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY was not found in the .env file."
    )


# Create the OpenAI client.
client = OpenAI(api_key=api_key)


print("=" * 70)
print("DOCUMENT Q&A")
print("=" * 70)


# ---------------------------------------------------------
# Step 1: Read the PDF
# ---------------------------------------------------------

print("\nLoading PDF...")

reader = PdfReader(PDF_PATH)

print(
    f"PDF loaded successfully. "
    f"Pages: {len(reader.pages)}"
)


# ---------------------------------------------------------
# Step 2: Extract text
# ---------------------------------------------------------

print("\nExtracting document text...")

document_text = ""

for page_number, page in enumerate(
    reader.pages,
    start=1,
):

    text = page.extract_text()

    if text:

        document_text += (
            f"\n--- Page {page_number} ---\n"
        )

        document_text += text


print("Document text extracted successfully.")


# ---------------------------------------------------------
# Step 3: Display extracted context
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("DOCUMENT CONTEXT")
print("-" * 70)

print(document_text)


# ---------------------------------------------------------
# Step 4: Ask a question
# ---------------------------------------------------------

question = "What is the total amount in the invoice?"


print("\n" + "-" * 70)
print("QUESTION")
print("-" * 70)

print(question)


# ---------------------------------------------------------
# Step 5: Send context + question to the LLM
# ---------------------------------------------------------

prompt = f"""
Answer the question using only the information
contained in the document below.

If the answer is not present in the document,
say that the information is not available.

DOCUMENT:
{document_text}

QUESTION:
{question}
"""


print("\nSending question to the LLM...")


response = client.responses.create(
    model="gpt-5-mini",
    input=prompt,
)


# ---------------------------------------------------------
# Step 6: Get the answer
# ---------------------------------------------------------

answer = response.output_text


print("\n" + "-" * 70)
print("ANSWER")
print("-" * 70)

print(answer)


print("\n" + "=" * 70)
print("DOCUMENT Q&A COMPLETED")
print("=" * 70)