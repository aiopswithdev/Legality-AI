# app/models.py

import json
import numpy as np
import faiss
import torch
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import (
    EMBED_MODEL_NAME,
    RERANKER_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    PRIMARY_EMBS_PATH,
    USE_GPU
)

# -------------------------------------------------
# Logger
# -------------------------------------------------

logger = logging.getLogger("models")
logger.setLevel(logging.INFO)

# -------------------------------------------------
# Global runtime objects
# -------------------------------------------------
# models.py
logger.info(f"models module loaded from: {__file__}")

embed_model = None
reranker = None
faiss_index = None
metadata = None
primary_embs = None


# -------------------------------------------------
# Device selection
# -------------------------------------------------

def get_device():
    if USE_GPU and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# -------------------------------------------------
# Model loading
# -------------------------------------------------

def load_embedding_model():
    global embed_model
    device = get_device()
    logger.info(f"Loading embedding model [{EMBED_MODEL_NAME}] on {device}...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    logger.info("Embedding model loaded successfully.")


def load_reranker():
    global reranker
    device = get_device()
    logger.info(f"Loading reranker model [{RERANKER_MODEL_NAME}] on {device}...")
    reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)
    logger.info("Reranker model loaded successfully.")


# -------------------------------------------------
# FAISS + metadata loading
# -------------------------------------------------

def load_faiss_index():
    global faiss_index
    logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    logger.info("FAISS index loaded successfully.")


def load_metadata_and_embeddings():
    global metadata, primary_embs

    logger.info(f"Loading metadata from {METADATA_PATH}...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]
        # Normalize label tokens to canonical form used by config (lowercase + underscores)
        for m in metadata:
            lbl = m.get("label", "")
            if isinstance(lbl, str):
                normalized = lbl.lower().replace("-", "_").replace(" ", "_")
                m["label"] = normalized

    logger.info(f"Loaded {len(metadata)} metadata records.")

    logger.info(f"Loading primary embeddings from {PRIMARY_EMBS_PATH}...")
    primary_embs = np.load(PRIMARY_EMBS_PATH)
    logger.info(f"Loaded primary embeddings: shape={primary_embs.shape}")

    assert len(metadata) == primary_embs.shape[0], (
        "Metadata and embedding count mismatch"
    )


# -------------------------------------------------
# Unified initializer
# -------------------------------------------------

def initialize_models():
    # logger.info("==== Initializing ML models & indexes ====")
    # load_embedding_model()
    # load_reranker()
    # load_faiss_index()
    # load_metadata_and_embeddings()
    # logger.info("==== Model initialization complete ====")
    logger.info("==== Initializing ML models & indexes ====")

    load_embedding_model()
    load_reranker()
    load_faiss_index()
    load_metadata_and_embeddings()

    # ----------------------------
    # Sanity checks (CRITICAL)
    # ----------------------------

    errors = []

    if embed_model is None:
        errors.append("embed_model is None")

    if reranker is None:
        errors.append("reranker is None")

    if faiss_index is None:
        errors.append("faiss_index is None")

    if metadata is None or len(metadata) == 0:
        errors.append("metadata is None or empty")

    if primary_embs is None:
        errors.append("primary_embs is None")
    else:
        logger.info(
            f"Primary embeddings shape: {primary_embs.shape}"
        )

    if errors:
        for err in errors:
            logger.error(f"[MODEL INIT ERROR] {err}")
        raise RuntimeError(
            "Model initialization failed. See errors above."
        )

    logger.info("✅ All models and indexes loaded successfully.")
    logger.info("==== Model initialization complete ====")
