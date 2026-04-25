# chunker.py
# Author: Frank Afelete Kofi Dogli | Index: 10022200146
# Part A: Chunking Strategy

"""
CHUNKING DESIGN DECISIONS
==========================

1. CSV (Election Results) → One chunk per row
   - Each row is a self-contained fact (candidate, party, votes, year)
   - Splitting a row would break the meaning
   - No overlap needed — rows are independent
   - Natural size: ~80-150 characters

2. PDF (Budget Statement) → Sliding window with overlap
   - Chunk size: 400 characters
       Why: Large enough to hold a full policy sentence,
            small enough to stay focused for retrieval
   - Overlap: 80 characters (20%)
       Why: Prevents a key sentence from being cut at a boundary
            and missed during retrieval
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    source: str        # "csv" or "pdf"
    text: str
    metadata: dict = field(default_factory=dict)


# ── CSV Chunker ────────────────────────────────────────────────────────────────
def _is_readable(text, threshold=0.6):
    """
    Returns False if a chunk is mostly garbled/scanned characters.
    Checks ratio of real words vs total tokens.
    """
    tokens = text.split()
    if len(tokens) < 5:
        return False
    readable = sum(1 for t in tokens if t.isalpha() and len(t) > 1)
    return (readable / len(tokens)) >= threshold

def chunk_csv_texts(texts):
    """One chunk per CSV row — each row is already a self-contained fact."""
    chunks = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        chunks.append(Chunk(
            chunk_id=f"csv_{i:05d}",
            source="csv",
            text=text,
            metadata={"row_index": i},
        ))
    return chunks


# ── PDF Chunker ────────────────────────────────────────────────────────────────

def chunk_pdf_texts(texts, chunk_size=1200, overlap=200):
    """
    Sliding window chunking for long PDF prose.
    Tries to break at sentence boundaries ('. ') to avoid
    cutting mid-sentence.
    """
    full_text = "\n\n".join(texts)

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(full_text):
        end = start + chunk_size

        # Snap to sentence boundary if possible
        if end < len(full_text):
            boundary = full_text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + overlap:
                end = boundary + 1

        snippet = full_text[start:end].strip()

        if len(snippet) > 20 and _is_readable(snippet):
            chunks.append(Chunk(
                chunk_id=f"pdf_{chunk_index:05d}",
                source="pdf",
                text=snippet,
                metadata={
                    "char_start": start,
                    "char_end": end,
                },
            ))
            chunk_index += 1

        start = end - overlap  # slide forward with overlap

    return chunks


# ── Combined ───────────────────────────────────────────────────────────────────

def build_all_chunks(csv_texts, pdf_texts):
    """Build and combine chunks from both data sources."""
    csv_chunks = chunk_csv_texts(csv_texts)
    pdf_chunks = chunk_pdf_texts(pdf_texts)
    all_chunks = csv_chunks + pdf_chunks
    print(f"CSV chunks: {len(csv_chunks)} | PDF chunks: {len(pdf_chunks)} | Total: {len(all_chunks)}")
    return all_chunks
