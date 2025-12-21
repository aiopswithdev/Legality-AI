import logging
from typing import List, Dict
import re

# import lexnlp.nlp.en.segments.sections as section_segmenter
from app.config import MIN_CLAUSE_LEN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Regex adapted directly from genesiscuad_ap.py (recall-first)
# ---------------------------------------------------------

CLAUSE_SPLIT_RE = re.compile(
    r"(?<=;)|"                # split after semicolon
    r"(?<=\.)|"               # split after period
    r"(?<=\))(?=\s+[A-Z])|"   # split after ) before capital letter
    r"(?<=\n)"                # split on newlines
)

# ---------------------------------------------------------
# Legal continuation / exception cues
# ---------------------------------------------------------

EXCEPTION_CUES = (
    "provided that",
    "except",
    "unless",
    "subject to",
    "notwithstanding",
    "so long as",
    "however",
    "including",
)

# MAX_CHARS = 1800   # deliberately high for recall
# MIN_CHARS = max(120, MIN_CLAUSE_LEN)
MIN_CHARS = MIN_CLAUSE_LEN


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"-\s+", "", text)  # PDF hyphenation
    return text.strip()

def _has_exception_prefix(text: str) -> bool:
    """
    Detect legal exception phrases that semantically bind forward.
    We check only the first ~80 chars to avoid false positives.
    """
    head = text.lower()[:80]
    return any(cue in head for cue in EXCEPTION_CUES)


def _merge_exceptions(blocks: List[str]) -> List[str]:
    """
    Merge legal exception fragments in BOTH directions:
    1) Forward merge: if fragment STARTS with exception cue
    2) Backward-aware merge: if fragment CONTAINS exception cue near start,
       merge it with the immediately following fragment
    """
    merged = []
    i = 0

    while i < len(blocks):
        cur = blocks[i].strip()
        if not cur:
            i += 1
            continue

        lower = cur.lower()

        # -------------------------------------------------
        # Case 1: backward-aware forward binding
        # e.g. "EXCEPT UNDER SECTION 11(a), IN NO EVENT..."
        # must bind to the following liability sentence
        # -------------------------------------------------
        if _has_exception_prefix(cur) and i + 1 < len(blocks):
            nxt = blocks[i + 1].strip()
            if nxt:
                merged.append(cur + " " + nxt)
                i += 2
                continue

        # -------------------------------------------------
        # Case 2: forward merge (existing behavior)
        # e.g. "provided that", "notwithstanding", etc.
        # -------------------------------------------------
        if merged and any(lower.startswith(c) for c in EXCEPTION_CUES):
            merged[-1] += " " + cur
        else:
            merged.append(cur)

        i += 1

    return merged



def _pack_clauses(clauses: List[str]) -> List[str]:
    """
    Pack only when a single clause is too short.
    NEVER merge two independent clauses.
    """
    chunks = []
    buf = ""

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue

        # If clause itself is long enough, flush buffer and keep it standalone
        if len(clause) >= MIN_CLAUSE_LEN:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            chunks.append(clause)
            continue

        # Clause is short → accumulate (exception prefixes, etc.)
        if not buf:
            buf = clause
        else:
            buf += " " + clause

    if buf.strip():
        chunks.append(buf.strip())

    return chunks




# ---------------------------------------------------------
# Page-level chunking
# ---------------------------------------------------------

def chunk_page_text(text: str, page_no: int) -> List[Dict]:
    text = _normalize(text)
    if not text:
        return []

    raw_clauses = CLAUSE_SPLIT_RE.split(text)
    # raw_clauses = [c.strip() for c in raw_clauses if len(c.strip()) >= MIN_CLAUSE_LEN]
    raw_clauses = [c.strip() for c in raw_clauses if c.strip()]
    # Merge legal exception fragments
    merged = _merge_exceptions(raw_clauses)

    # Pack clauses (recall-first)
    chunks = _pack_clauses(merged)

    return [
        {
            "page_no": page_no,
            "clause_text": c
        }
        for c in chunks
        if len(c) >= MIN_CLAUSE_LEN
    ]


# ---------------------------------------------------------
# Document-level API (USED BY PIPELINE)
# ---------------------------------------------------------

def chunk_pages(pages: List[Dict]) -> List[Dict]:
    all_chunks = []

    for page in pages:
        page_chunks = chunk_page_text(
            page.get("text", ""),
            page.get("page_no")
        )
        all_chunks.extend(page_chunks)

    logger.info(f"Chunked document into {len(all_chunks)} semantic clauses")
    return all_chunks


# ---------------------------------------------------------
# Deduplication (keep — same as before)
# ---------------------------------------------------------

def _dedup_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []

    for c in chunks:
        key = _dedup_key(c["clause_text"])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique
