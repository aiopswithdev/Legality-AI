# app/config.py

# ============================================================
# Model configuration
# ============================================================

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"

# ============================================================
# Runtime / hardware
# ============================================================

# Set to False if deploying CPU-only
USE_GPU = True

# ============================================================
# Vector DB / data paths
# ============================================================

FAISS_INDEX_PATH = r"FAISS/clauses.index"
METADATA_PATH = r"FAISS/metadata.jsonl"
PRIMARY_EMBS_PATH = r"FAISS/primary_embs.npy"
# FAISS_INDEX_PATH = r"C:\Users\admin\OneDrive\Documents\Genesis\FAISS\clauses.index"
# METADATA_PATH = r"C:\Users\admin\OneDrive\Documents\Genesis\FAISS\metadata.jsonl"
# PRIMARY_EMBS_PATH = r"C:\Users\admin\OneDrive\Documents\Genesis\FAISS\primary_embs.npy"

# ============================================================
# Retrieval configuration
# ============================================================

# Number of FAISS candidates retrieved
TOP_K_RETRIEVAL = 25

# Number of candidates reranked by cross-encoder
TOP_K_RERANK = 10
RERANKER_BATCH_SIZE = 32  # or even 64

# ============================================================
# Scoring weights (matches your notebook logic)
# ============================================================

WEIGHTS = {
    "identity": 0.5,
    "semantic": 0.4,
    "margin": 0.1
}

# ============================================================
# Clause length constraints (used in chunking)
# ============================================================

MIN_CLAUSE_LEN = 40

# ============================================================
# Risk thresholds (used OUTSIDE scoring.py)
# ============================================================

RISK_THRESHOLDS = {
    "termination_for_convenience": (0.60, 0.75),
    "non_compete": (0.58, 0.72),
    "uncapped_liability": (0.60, 0.70),
    "_default": (0.60, 0.70)
}

# ============================================================
# Confidence band labels (UI-facing)
# ============================================================

RISK_BANDS = {
    "LOW": "low",
    "REVIEW": "review",
    "HIGH": "high"
}

# Minimum total extracted characters required to SKIP OCR
# If extracted text is shorter than this, OCR fallback is used
OCR_MIN_CHARS = 200

# DPI for rendering PDF pages before OCR
# Higher = better OCR accuracy, slower performance
OCR_RESOLUTION = 300