# app/pipeline.py

from collections import defaultdict
from typing import Dict, List
import re

import logging
from app.document_io import extract_pages_from_pdf_bytes
from app.chunking import chunk_pages, deduplicate_chunks
from app.scoring import score_clauses_batch
from app.config import RISK_THRESHOLDS, RISK_BANDS, WEIGHTS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------
# Label semantic gates
# -----------------------------

_NON_COMPETE_RESTRICTION_VERBS = {
    "will not", "shall not", "may not",
    "not enter into", "agrees not to",
    "cannot", "prohibited", "restricted"
}

_NON_COMPETE_COMPETITION_TERMS = {
    "competitor", "competitors", "competing", "competitive", "products",
    "employment", "engage", "operate",
    "ownership", "interest",
    "promote", "promotion", "advertising", "display",
    "sales", "market"
}


def _passes_non_compete_gate(text: str) -> bool:
    """
    Legal invariant:
    A non-compete must restrict ECONOMIC COMPETITION,
    not merely speech or conduct.
    """
    t = text.lower()
    return (
        any(v in t for v in _NON_COMPETE_RESTRICTION_VERBS)
        and
        any(c in t for c in _NON_COMPETE_COMPETITION_TERMS)
    )

# ---------------------------------------------------------
# Per-label risk band assignment
# ---------------------------------------------------------

def assign_label_band(
    semantic_score: float,
    identity: float,
    label: str
) -> str:
    """
    Assign LOW / REVIEW / HIGH_ for a specific label.
    """
    label = label.lower().replace("-", "_").replace(" ", "_")
    if identity >= 0.98:
        return RISK_BANDS["HIGH"]
    semantic_score = max(0.0, min(1.0, semantic_score))

    # low, high = RISK_THRESHOLDS.get(label, RISK_THRESHOLDS["_default"])
    # if label not in RISK_THRESHOLDS:
    #     logger.warning(f"Unknown label '{label}' — using default risk thresholds")
    if label in RISK_THRESHOLDS:
        low, high = RISK_THRESHOLDS[label]
    else:
        low, high = RISK_THRESHOLDS["_default"]
        logger.warning(f"Unknown label '{label}' — using default risk thresholds")

    if semantic_score >= high:
        return RISK_BANDS["HIGH"]
    elif semantic_score >= low:
        return RISK_BANDS["REVIEW"]
    else:
        return RISK_BANDS["LOW"]


# ---------------------------------------------------------
# Extract multi-label signals from scoring output
# ---------------------------------------------------------

def extract_clause_labels(score_out: Dict) -> List[Dict]:
    """
    Convert top_matches into per-label signals.
    Each clause may map to multiple labels.
    """
    label_to_score = {}

    for match in score_out["top_matches"]:
        raw_label = match["label"]
        label = raw_label.lower().replace("-", "_").replace(" ", "_")
        score = match["score"]

        label_to_score[label] = max(label_to_score.get(label, 0.0), score)

    labels = []
    for label, semantic_score in label_to_score.items():
        final_score = (
            WEIGHTS["identity"] * score_out["identity"] +
            WEIGHTS["semantic"] * semantic_score +
            WEIGHTS["margin"] * score_out["margin"]
        )

        band = assign_label_band(
            semantic_score=semantic_score,
            identity=score_out["identity"],
            label=label
        )

        labels.append({
            "label": label,
            "semantic_score": float(semantic_score),
            "final_score": float(final_score),
            "band": band
        })

    labels.sort(key=lambda x: x["final_score"], reverse=True)
    return labels


# ---------------------------------------------------------
# Clause-level analysis (multi-label)
# ---------------------------------------------------------

# def analyze_clauses(chunks: List[Dict]) -> List[Dict]:
#     """
#     Score each clause and attach multi-label risk results.
#     """
#     results = []
#     # In analyze_clauses
#     texts = [c["clause_text"] for c in chunks]
#     batch_results = score_clauses_batch(texts)
#     for chunk, score_out in zip(chunks, batch_results):
#         clause_text = chunk["clause_text"]
#         page_no = chunk["page_no"]
#         if not score_out:
#             continue

#         labels = extract_clause_labels(score_out)

#         # Skip clauses that are SAFE for all labels
#         if all(l["band"] == RISK_BANDS["SAFE"] for l in labels):
#             continue

#         results.append({
#             "page_no": page_no,
#             "clause_text": clause_text,
#             "labels": labels,
#             "identity": score_out["identity"],
#             "semantic": score_out["semantic"],
#             "margin": score_out["margin"],
#             "top_matches": score_out["top_matches"]
#         })

#     return results

# helper: normalize clause text for dedup/merge
def _normalize_clause_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().lower()
    # optionally remove numbers/enum markers if desired
    return s

# helper: merge labels for same clause (take max final_score per label)
def _merge_labels(label_lists: List[List[Dict]]) -> List[Dict]:
    merged = {}
    for lbls in label_lists:
        for l in lbls:
            key = l["label"]
            cur = merged.get(key)
            if cur is None or l["final_score"] > cur["final_score"]:
                merged[key] = {
                    "label": key,
                    "semantic_score": float(l["semantic_score"]),
                    "final_score": float(l["final_score"]),
                    "band": l["band"]
                }
    out = list(merged.values())
    out.sort(key=lambda x: x["final_score"], reverse=True)
    return out

# Notebook-faithful analyze_clauses with post-scoring label filtering + optional dedup
def analyze_clauses(chunks: List[Dict], dedup: bool = True) -> List[Dict]:
    """
    1) Score all clauses (batch)
    2) Extract per-label signals and compute final scores (extract_clause_labels)
    3) Keep ONLY labels with band != SAFE (notebook behaviour)
    4) Optionally deduplicate/merge near-identical clauses (eval_pdf_unique_clauses style)
    """
    texts = [c["clause_text"] for c in chunks]
    batch_results = score_clauses_batch(texts)

    # collect raw clause outputs
    raw_clauses = []
    for chunk, score_out in zip(chunks, batch_results):
        if score_out is None:
            continue

        labels_all = extract_clause_labels(score_out)  # returns per-label dicts with final_score and band

        # --------------------------------------------------
        # NON-COMPETE SEMANTIC GATE (POST-SCORING)
        # --------------------------------------------------
        filtered_labels = []
        for l in labels_all:
            if l["label"] == "non_compete":
                if not _passes_non_compete_gate(chunk["clause_text"]):
                    continue  # ❌ drop false non-compete (e.g. non-disparagement)
            filtered_labels.append(l)

        labels_all = filtered_labels
        # Notebook: we only surface labels that are not SAFE/LOW
        risky_labels = [l for l in labels_all if l["band"] != RISK_BANDS["LOW"]]

        # keep the clause only if there is at least one risky label
        if len(risky_labels) == 0:
            # notebook would still compute these but when final output it filters to non-LOW.
            # To mimic exact notebook "eval_pdf_unique_clauses" behavior: skip non-risky clauses.
            continue

        raw_clauses.append({
            "page_no": chunk["page_no"],
            "clause_text": chunk["clause_text"],
            "final_score": score_out["final_score"],
            "identity": score_out["identity"],
            "semantic": score_out["semantic"],
            "margin": score_out["margin"],
            # "top_matches": score_out["top_matches"],
            "labels": risky_labels
        })

    if not dedup:
        return raw_clauses

    # Deduplicate/merge similar clauses (eval_pdf_unique_clauses)
    grouped = defaultdict(list)  # normalized_text -> list of clause dicts
    for rc in raw_clauses:
        key = _normalize_clause_text(rc["clause_text"])
        grouped[key].append(rc)

    final_clauses = []
    for key, group in grouped.items():
        # choose representative (you can pick first or highest final_score across labels)
        # We'll merge labels across group and pick page_no from the first item
        all_label_lists = [g["labels"] for g in group]
        merged_labels = _merge_labels(all_label_lists)

        # pick representative clause (first)
        rep = group[0]
        # Calculate clause-level final_score as max of original clause final_scores from the group
        clause_final_score = max((g.get("final_score", 0.0) for g in group), default=rep.get("final_score", 0.0))
        
        final_clauses.append({
            "page_no": rep["page_no"],
            "clause_text": rep["clause_text"],
            "final_score": clause_final_score,
            "labels": merged_labels,
            "identity": rep["identity"],
            "semantic": rep["semantic"],
            "margin": rep["margin"]
            # "top_matches": rep["top_matches"]
        })

    # optional: sort final_clauses by descending highest label final_score
    final_clauses.sort(key=lambda c: max((l["final_score"] for l in c["labels"]), default=0.0), reverse=True)

    return final_clauses

# def analyze_clauses(chunks: List[Dict]) -> List[Dict]:
#     """
#     Notebook-faithful clause analysis:
#     - NO SAFE pruning
#     - Every clause is returned
#     - Identity is a signal, not a filter
#     """
#     results = []

#     texts = [c["clause_text"] for c in chunks]
#     batch_results = score_clauses_batch(texts)

#     for chunk, score_out in zip(chunks, batch_results):
#         if score_out is None:
#             continue

#         labels = extract_clause_labels(score_out)

#         # 🔴 NOTEBOOK BEHAVIOR:
#         # Do NOT drop SAFE clauses
#         # Every clause survives to output

#         results.append({
#             "page_no": chunk["page_no"],
#             "clause_text": chunk["clause_text"],
#             "labels": labels,
#             "identity": score_out["identity"],
#             "semantic": score_out["semantic"],
#             "margin": score_out["margin"],
#             "top_matches": score_out["top_matches"]
#         })

#     return results


# ---------------------------------------------------------
# Document-level aggregation (multi-label aware)
# ---------------------------------------------------------

# def aggregate_document_risk(clauses: List[Dict]) -> Dict:
#     """
#     Aggregate clause-level multi-label risk
#     into per-label and document-level summaries.
#     """
#     per_label = defaultdict(list)

#     for clause in clauses:
#         for lbl in clause["labels"]:
#             per_label[lbl["label"]].append(lbl)

#     label_summary = {}
#     document_band = RISK_BANDS["LOW"]

#     for label, items in per_label.items():
#         max_score = max(i["semantic_score"] for i in items)
#         high_risk_count = sum(
#             i["band"] == RISK_BANDS["HIGH"] for i in items
#         )

#         label_summary[label] = {
#             "max_score": max_score,
#             "high_risk_clauses": high_risk_count,
#             "total_clauses": len(items)
#         }

#         # 🔧 FIX STARTS HERE
#         label_norm = label.lower().replace("-", "_").replace(" ", "_")

#         if label_norm in RISK_THRESHOLDS:
#             low, high = RISK_THRESHOLDS[label_norm]
#         else:
#             low, high = RISK_THRESHOLDS["_default"]
#             logger.warning(
#                 f"Unknown label '{label_norm}' during document aggregation — using default thresholds"
#             )
#         # 🔧 FIX ENDS HERE

#         if max_score >= high:
#             document_band = RISK_BANDS["HIGH"]
#         elif max_score >= low and document_band != RISK_BANDS["HIGH"]:
#             document_band = RISK_BANDS["REVIEW"]

#     return {
#         "document_risk": document_band,
#         "label_summary": label_summary
#     }

def aggregate_document_risk(clauses: List[Dict]) -> Dict:
    """
    Aggregate clause-level risks into document-level summary.
    Open-set label safe (notebook-faithful).
    """
    label_summary = {}

    for clause in clauses:
        for lbl in clause.get("labels", []):
            label = lbl["label"]
            score = lbl["final_score"]

            label_norm = label.lower().replace("-", "_").replace(" ", "_")

            if label_norm not in label_summary:
                label_summary[label_norm] = {
                    "max_score": score,
                    "high_risk_clauses": 0,
                    "total_clauses": 0
                }
            else:
                label_summary[label_norm]["max_score"] = max(
                    label_summary[label_norm]["max_score"], score
                )

            label_summary[label_norm]["total_clauses"] += 1

            # Threshold-safe banding
            if label_norm in RISK_THRESHOLDS:
                _, high = RISK_THRESHOLDS[label_norm]
            else:
                _, high = RISK_THRESHOLDS["_default"]
                logger.warning(
                    f"Unknown label '{label_norm}' during document aggregation — using default thresholds"
                )

            if score >= high:
                label_summary[label_norm]["high_risk_clauses"] += 1

    # Overall document risk (simple heuristic)
    document_risk = "low_risk"
    for v in label_summary.values():
        if v["high_risk_clauses"] > 0:
            document_risk = "high_risk"
            break

    return {
        "document_risk": document_risk,
        "label_summary": label_summary
    }


# ---------------------------------------------------------
# Main pipeline entrypoint
# ---------------------------------------------------------

def analyze_document(pdf_bytes: bytes) -> Dict:
    from app.models import embed_model
    import logging

    logger = logging.getLogger(__name__)
    if embed_model is None:
        logger.error("embed model is None at pipeline entry")
    else:
        logger.info("embed model is AVAILABLE at pipeline entry")
    """
    End-to-end production pipeline.

    Input:
        pdf_bytes (bytes)

    Output (JSON-serializable):
        {
          document_risk,
          label_summary,
          clauses
        }
    """
    
    # 1. Extract page-wise text
    pages = extract_pages_from_pdf_bytes(pdf_bytes)

    # 2. Chunk into clauses
    nodedup = chunk_pages(pages)
    chunks = dedup = deduplicate_chunks(nodedup)
    logger.info(f"pages={len(pages)}, chunks_before_dedup={len(nodedup)}, chunks_after_dedup={len(dedup)}")

    # 3. Multi-label clause scoring
    clause_results = analyze_clauses(chunks)

    # 4. Aggregate document risk
    doc_summary = aggregate_document_risk(clause_results)

    # 5. Calculate doc_score: average of all clause final_scores * 10, rounded
    if clause_results:
        avg_final_score = sum(c.get("final_score", 0.0) for c in clause_results) / len(clause_results)
        doc_score = round(avg_final_score * 10)
    else:
        doc_score = 0

    return {
        "document_risk": doc_summary["document_risk"],
        "doc_score": doc_score,
        "label_summary": doc_summary["label_summary"],
        "clauses": clause_results
    }
