# app/scoring.py
from typing import List
import numpy as np
from scipy.special import expit  # sigmoid
import logging
import inspect
# import app.models as models

# from app.models import (
#     embed_model,
#     reranker,
#     faiss_index,
#     metadata,
#     primary_embs
# )
from app.config import (
    TOP_K_RETRIEVAL,
    TOP_K_RERANK,
    RERANKER_BATCH_SIZE,
    WEIGHTS
)
import app.models as models

# print("SCORING sees models at:", inspect.getfile(models))
# ----------------------------
# Embedding helpers
# ----------------------------
logger = logging.getLogger("scoring")
logger.setLevel(logging.INFO)
# logger.info(f"[DEBUG] scoring.py sees models from: {models.embed_model}")

# -------------------------------------------------
# Non-compete recall anchors (retrieval only)
# -------------------------------------------------

# _NON_COMPETE_ANCHORS = {
#     "ownership interest",
#     "perform services for",
#     "engage in or perform",
#     "engage in",
#     "services for any competitor",
#     "any competitor",
#     "following termination",
#     "post termination",
#     "after termination",
# }

# def _contains_non_compete_anchor(text: str) -> bool:
#     t = text.lower()
#     return any(anchor in t for anchor in _NON_COMPETE_ANCHORS)

def embed_clause(text: str) -> np.ndarray:
    primary = models.embed_model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )  # shape (1, dim)

    secondary = models.embed_model.encode(
        ["CONTEXT:\n" + text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )  # shape (1, dim)

    return np.hstack([primary, secondary]).astype("float32")  # shape (1, dim*2)

# ----------------------------
# Identity score
# ----------------------------

def compute_identity_score(query_primary_emb: np.ndarray) -> float:
    """
    Identity = max cosine similarity against
    stored primary embeddings.
    """
    sims = np.dot(models.primary_embs, query_primary_emb.T).squeeze()
    return float(np.max(sims))


# ----------------------------
# Semantic retrieval + rerank
# ----------------------------

def semantic_retrieval(query_vec: np.ndarray):
    """
    FAISS dense retrieval.
    """
    scores, indices = models.faiss_index.search(
        query_vec,
        TOP_K_RETRIEVAL
    )
    return indices[0]


def rerank(query_text: str, candidate_indices):
    """
    Cross-encoder reranking.
    """
    pairs = [
        [query_text, models.metadata[idx]["answer_text"]]
        for idx in candidate_indices
    ]

    raw_scores = models.reranker.predict(
        pairs,
        batch_size=TOP_K_RERANK
    )

    semantic_scores = expit(raw_scores)

    return semantic_scores


# ----------------------------
# Margin score
# ----------------------------

def compute_margin(scores: np.ndarray) -> float:
    """
    Margin = top1 - top2 semantic score.
    """
    if len(scores) < 2:
        return 0.0

    sorted_scores = np.sort(scores)[::-1]
    return float(sorted_scores[0] - sorted_scores[1])


# ----------------------------
# Main scoring function
# ----------------------------

def score_clause(text: str) -> dict:
    # ----------------------------
    # Runtime safety checks
    # ----------------------------
    if models.embed_model is None:
        raise RuntimeError("embed_model is None at scoring time")

    if models.reranker is None:
        raise RuntimeError("reranker is None at scoring time")

    if models.faiss_index is None:
        raise RuntimeError("faiss_index is None at scoring time")

    if models.metadata is None:
        raise RuntimeError("metadata is None at scoring time")

    if models.primary_embs is None:
        raise RuntimeError("primary_embs is None at scoring time")

    # ---- rest of scoring logic ----

    # ---- 1. Embed ----
    query_vec = embed_clause(text)

    # ---- 2. Identity ----
    identity = compute_identity_score(query_vec[:, :models.primary_embs.shape[1]])

    # ---- 3. FAISS retrieval ----
    candidate_indices = semantic_retrieval(query_vec)

    # ---- 4. Cross-encoder rerank ----
    semantic_scores = rerank(text, candidate_indices)

    # ---- 5. Margin ----
    margin = compute_margin(semantic_scores)

    # ---- 6. Final score ----
    final_score = (
        WEIGHTS["identity"] * identity +
        WEIGHTS["semantic"] * float(np.max(semantic_scores)) +
        WEIGHTS["margin"]   * margin
    )

    # ---- 7. Build matches ----
    top_matches = []
    for idx, s in zip(candidate_indices, semantic_scores):
        m = models.metadata[idx]
        top_matches.append({
            "index_id": idx,
            "score": float(s),
            "label": m["label"],
            "answer_text": m["answer_text"],
            "source_title": m["source_title"]
        })

    top_matches.sort(key=lambda x: x["score"], reverse=True)

    return {
        "final_score": float(final_score),
        "identity": float(identity),
        "semantic": float(np.max(semantic_scores)),
        "margin": float(margin),
        "top_matches": top_matches[:TOP_K_RERANK]
    }
# ----------------------------
# Sub-clause probing helpers
# ----------------------------

def _make_subclauses(text: str, window: int = 240, stride: int = 120) -> List[str]:
    """
    Generate overlapping sub-clause windows for long legal clauses.
    Used ONLY for retrieval probing, never returned.
    """
    text = text.strip()
    if len(text) <= window:
        return [text]

    subs = []
    for i in range(0, max(len(text) - window + 1, 1), stride):
        subs.append(text[i:i + window])
    return subs

def score_clauses_batch(texts: List[str]) -> List[dict]:
    logger.info(f"RERANKER INVOKED for {len(texts)} clauses")
    
    """
    Batch score multiple clauses (vectorized).
    Returns list of score_out dicts matching score_clause's output format.
    """
    if models.embed_model is None or models.faiss_index is None or models.reranker is None:
        raise RuntimeError("models not initialized for batch scoring")

    n = len(texts)
    # 1) Embed primary and secondary in batches (convert_to_numpy True)
    primary_embs_q = models.embed_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32
    ).astype("float32")   # (n, d)
    secondary_embs_q = models.embed_model.encode(
        ["CONTEXT:\n" + t for t in texts],
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32
    ).astype("float32")   # (n, d)

    # Combined vector for FAISS search
    q_combined = np.hstack([primary_embs_q, secondary_embs_q]).astype("float32")  # (n, 2d)

    # 2) FAISS retrieval (batched)
    
    id_sims = np.dot(primary_embs_q, models.primary_embs.T)  # (n, M)
    identity_scores = np.max(id_sims, axis=1).astype(float)  # (n,)

    # D, I = models.faiss_index.search(q_combined, TOP_K_RETRIEVAL)  # I shape (n, k)
    # candidate_indices_per_query = I.tolist()

    # ---------------------------------------------------------
    # Sub-clause probing for retrieval (LONG clauses only)
    # ---------------------------------------------------------

    candidate_indices_per_query = []

    for qi, text in enumerate(texts):
        # Generate probing texts
        probes = _make_subclauses(text)

        # Embed probes
        probe_primary = models.embed_model.encode(
            probes,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        probe_secondary = models.embed_model.encode(
            ["CONTEXT:\n" + p for p in probes],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        probe_vecs = np.hstack([probe_primary, probe_secondary]).astype("float32")

        # FAISS search for each probe
        probe_candidates = set()
        for pv in probe_vecs:
            _, idxs = models.faiss_index.search(
                pv.reshape(1, -1),
                TOP_K_RETRIEVAL
            )
            probe_candidates.update(idxs[0].tolist())

        candidate_indices_per_query.append(list(probe_candidates))

    # 3) Build reranker pairs for all (query, candidate.answer_text)
    # all_pairs = []
    # idx_map = []  # maps pair index -> (query_idx, candidate_idx)
    # for qi, candidate_indices in enumerate(candidate_indices_per_query):
    #     for idx in candidate_indices:
    #         all_pairs.append([texts[qi], models.metadata[idx]["answer_text"]])
    #         idx_map.append((qi, idx))

    # if len(all_pairs) == 0:
    #     return [None] * n

    all_pairs = []
    idx_map = []
    precomputed_matches = {qi: [] for qi in range(n)}

    for qi, candidate_indices in enumerate(candidate_indices_per_query):
    # CASE 1: Very high identity → rerank ONLY top-1 (guaranteed at least one)
        # if identity_scores[qi] >= 0.98:
        #     idx = candidate_indices[0]  # top-1 ONLY
        #     all_pairs.append([texts[qi], models.metadata[idx]["answer_text"]])
        #     idx_map.append((qi, idx))
        #     continue

        # Otherwise, send to reranker
        for idx in candidate_indices:
            all_pairs.append([texts[qi], models.metadata[idx]["answer_text"]])
            idx_map.append((qi, idx))
    
    if len(all_pairs) == 0:
        return [None] * n

    # 4) Cross-encoder predict (batched by reranker)
    raw_scores = models.reranker.predict(all_pairs, batch_size=RERANKER_BATCH_SIZE)
    semantic_scores_all = expit(raw_scores)  # shape (len(all_pairs),)

    # 5) Scatter back semantic scores per query
    semantic_per_query = [[] for _ in range(n)]
    # top_matches_per_query = [[] for _ in range(n)]
    top_matches_per_query = [precomputed_matches[i] for i in range(n)]
    for s, (qi, idx) in zip(semantic_scores_all, idx_map):
        semantic_per_query[qi].append((idx, float(s)))

    # sort matches per query and build top_matches
    for qi, lst in enumerate(semantic_per_query):
        lst.sort(key=lambda x: x[1], reverse=True)
        for idx, s in lst[:TOP_K_RERANK]:
            m = models.metadata[idx]
            top_matches_per_query[qi].append({
                "index_id": int(idx),
                "score": float(s),
                "label": m["label"],
                "answer_text": m["answer_text"],
                "source_title": m.get("source_title")
            })

    # 6) Compute identity scores vectorized using primary_embs stored (models.primary_embs)
    # primary_embs_q shape (n, d). models.primary_embs shape (M, d)
    # id_sims = np.dot(primary_embs_q, models.primary_embs.T)  # (n, M)
    # identity_scores = np.max(id_sims, axis=1).astype(float)  # (n,)

    # 7) Margin + final score and assemble output
    outputs = []
    for qi in range(n):
        sem_scores = [m["score"] for m in top_matches_per_query[qi]]
        if len(sem_scores) == 0:
            outputs.append(None)
            continue
        semantic = float(max(sem_scores))
        margin = float((sem_scores[0] - sem_scores[1]) if len(sem_scores) > 1 else 0.0)
        final_score = (
            WEIGHTS["identity"] * float(identity_scores[qi]) +
            WEIGHTS["semantic"] * semantic +
            WEIGHTS["margin"] * margin
        )

        outputs.append({
            "final_score": float(final_score),
            "identity": float(identity_scores[qi]),
            "semantic": float(semantic),
            "margin": float(margin),
            "top_matches": top_matches_per_query[qi][:TOP_K_RERANK]
        })

    return outputs
