# app/document_io.py

import io
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import logging
from app.config import OCR_MIN_CHARS, OCR_RESOLUTION

logger = logging.getLogger(__name__)

# ------------------------------------------------
# Utility: check text quality
# ------------------------------------------------

def _is_text_usable(text: str, min_chars: int = OCR_MIN_CHARS) -> bool:
    if not text:
        return False
    return len(text.strip()) >= min_chars


# ------------------------------------------------
# Primary extraction: embedded PDF text
# ------------------------------------------------

def _extract_with_pdfplumber(pdf_bytes: bytes):
    pages = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({
                "page_no": i + 1,
                "text": text
            })

    return pages


# ------------------------------------------------
# OCR fallback extraction
# ------------------------------------------------
import re

def normalize_ocr_text(text: str) -> str:
    # 1. Fix hyphenated line breaks (OCR classic)
    # Example: "com-\npetition" → "competition"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 2. Normalize Windows/Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Collapse single newlines INSIDE paragraphs
    # Keep double newlines (paragraph boundaries)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 4. Normalize excessive newlines (3+ → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()

def _extract_with_ocr(pdf_bytes: bytes):
    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=OCR_RESOLUTION)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text = pytesseract.image_to_string(
            img,
            config="--oem 3 --psm 6"
        )

        text = normalize_ocr_text(text)

        pages.append({
            "page_no": i + 1,
            "text": text
        })

    return pages
# def _extract_with_ocr(pdf_bytes: bytes):
#     pages = []

#     doc = fitz.open(stream=pdf_bytes, filetype="pdf")

#     for i, page in enumerate(doc):
#         pix = page.get_pixmap(dpi=OCR_RESOLUTION)
#         img = Image.frombytes(
#             "RGB",
#             [pix.width, pix.height],
#             pix.samples
#         )

#         text = pytesseract.image_to_string(img)

#         pages.append({
#             "page_no": i + 1,
#             "text": text
#         })

#     return pages


# ------------------------------------------------
# Public API
# ------------------------------------------------

def extract_pages_from_pdf_bytes(pdf_bytes: bytes):
    """
    Main entry point used by the pipeline.

    Returns:
    [
      {"page_no": int, "text": str},
      ...
    ]
    """
    
    # --- Try text-based extraction first ---
    pages = _extract_with_pdfplumber(pdf_bytes)
    full_text = " ".join(p["text"] for p in pages)
    total_chars = len(full_text)
    logger.info(f"pdfplumber extracted total chars: {total_chars}")
    if _is_text_usable(full_text):
        return pages
    logger.info("Falling back to OCR due to low extracted text.")

    if _is_text_usable(full_text):
        return pages

    # --- Fallback to OCR ---
    return _extract_with_ocr(pdf_bytes)
