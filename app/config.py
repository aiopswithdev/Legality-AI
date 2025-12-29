# app/config.py

import os
from pathlib import Path
from typing import List

# ============================================================
# Environment Variable Support
# ============================================================

def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_env_list(key: str, default: List[str]) -> List[str]:
    """Get list from comma-separated environment variable."""
    value = os.getenv(key, "")
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


# ============================================================
# Model configuration
# ============================================================

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-en-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")

# ============================================================
# Runtime / hardware
# ============================================================

# Set to False if deploying CPU-only
USE_GPU = get_env_bool("USE_GPU", True)

# ============================================================
# Vector DB / data paths
# ============================================================

# Default paths (relative to project root)
DEFAULT_FAISS_DIR = os.getenv("FAISS_DIR", "FAISS")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.join(DEFAULT_FAISS_DIR, "clauses.index"))
METADATA_PATH = os.getenv("METADATA_PATH", os.path.join(DEFAULT_FAISS_DIR, "metadata.jsonl"))
PRIMARY_EMBS_PATH = os.getenv("PRIMARY_EMBS_PATH", os.path.join(DEFAULT_FAISS_DIR, "primary_embs.npy"))

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
OCR_RESOLUTION = int(os.getenv("OCR_RESOLUTION", "300"))

# ============================================================
# API Configuration
# ============================================================

# CORS settings
CORS_ORIGINS = get_env_list("CORS_ORIGINS", ["*"])
CORS_ALLOW_CREDENTIALS = get_env_bool("CORS_ALLOW_CREDENTIALS", True)
CORS_ALLOW_METHODS = get_env_list("CORS_ALLOW_METHODS", ["*"])
CORS_ALLOW_HEADERS = get_env_list("CORS_ALLOW_HEADERS", ["*"])

# File upload settings
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")