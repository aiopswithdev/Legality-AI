import fitz  # PyMuPDF
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def highlight_clauses_in_pdf(
    pdf_bytes: bytes,
    clauses: List[Dict]
) -> bytes:
    """
    Given original PDF bytes and analyze() response clauses,
    return a new PDF with highlighted clauses.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for clause in clauses:
        page_no = clause.get("page_no")
        text = clause.get("clause_text")

        if not page_no or not text:
            continue

        page_index = page_no - 1
        if page_index < 0 or page_index >= len(doc):
            continue

        page = doc[page_index]

        # Primary search
        matches = page.search_for(text)

        # Fallback: search first 200 chars if exact match fails
        if not matches:
            snippet = text[:200]
            matches = page.search_for(snippet)

        for rect in matches:
            annot = page.add_highlight_annot(rect)
            annot.set_info(
                title="Legality-AI",
                content="Detected risky clause"
            )
            annot.update()

    output = doc.tobytes()
    doc.close()
    return output
