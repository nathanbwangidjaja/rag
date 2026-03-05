"""
PDF text extraction using PyMuPDF.

Considerations:
- PyMuPDF handles most PDF layouts well, including multi-column and embedded fonts.
- We extract per-page to preserve document structure and enable page-level citations.
- Some PDFs have text baked into images (scanned docs) -- we don't handle OCR here
  but that would be a natural extension (e.g. pytesseract).
- Headers/footers often repeat across pages. We do a simple heuristic to strip
  lines that appear on most pages, since they add noise to retrieval.
"""

import fitz  # pymupdf
from collections import Counter


def extract_pages(pdf_path: str) -> list[dict]:
    """Pull text out of each page. Returns list of {page, text}."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    doc.close()
    return pages


def _find_repeated_lines(pages: list[dict], threshold=0.6) -> set[str]:
    """Find lines that show up on most pages -- likely headers/footers."""
    if len(pages) < 3:
        return set()

    line_counts = Counter()
    for p in pages:
        unique_lines = set(p["text"].splitlines())
        for line in unique_lines:
            cleaned = line.strip()
            if len(cleaned) > 2:  # skip blank-ish lines
                line_counts[cleaned] += 1

    cutoff = len(pages) * threshold
    return {line for line, count in line_counts.items() if count >= cutoff}


def clean_pages(pages: list[dict]) -> list[dict]:
    """Remove repeated headers/footers and normalize whitespace."""
    repeated = _find_repeated_lines(pages)

    cleaned = []
    for p in pages:
        lines = p["text"].splitlines()
        filtered = [l for l in lines if l.strip() not in repeated]
        text = "\n".join(filtered).strip()
        # collapse multiple blank lines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        if text:
            cleaned.append({"page": p["page"], "text": text})

    return cleaned


def parse_pdf(pdf_path: str) -> list[dict]:
    """Main entry: extract + clean pages from a PDF."""
    pages = extract_pages(pdf_path)
    return clean_pages(pages)
