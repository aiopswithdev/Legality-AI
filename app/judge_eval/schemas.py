from pydantic import BaseModel
from typing import Literal


class JudgeInput(BaseModel):
    # ---- Identity ----
    clause_id: str
    analysis_id: str

    # ---- Core content ----
    clause_text: str
    label: str

    # ---- Scores (what the judge audits) ----
    identity: float
    semantic_score: float
    margin: float
    final_score: float
    band: Literal["low", "review", "high"]

    # ---- Threshold context ----
    threshold_low: float
    threshold_high: float

    # ---- Versioning ----
    threshold_version: str
