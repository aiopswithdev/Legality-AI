"""
Pydantic schemas for request/response validation and API documentation.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# -------------------------------------------------
# Response Models
# -------------------------------------------------

class LabelRisk(BaseModel):
    """Risk assessment for a specific label."""
    label: str = Field(..., description="The risk label identifier")
    semantic_score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity score")
    final_score: float = Field(..., ge=0.0, le=1.0, description="Weighted final risk score")
    band: str = Field(..., description="Risk band: 'low', 'review', or 'high'")


class ClauseResult(BaseModel):
    """Analysis result for a single clause."""
    page_no: int = Field(..., description="Page number where clause was found")
    clause_text: str = Field(..., description="The extracted clause text")
    labels: List[LabelRisk] = Field(..., description="Risk assessments for this clause")
    final_score: float = Field(..., ge=0.0, le=1.0, description="Weighted final risk score")
    identity: float = Field(..., ge=0.0, le=1.0, description="Identity match score")
    semantic: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity score")
    margin: float = Field(..., description="Margin score")
    # top_matches: List[Dict[str, Any]] = Field(..., description="Top matching reference clauses")


class LabelSummary(BaseModel):
    """Summary statistics for a risk label across the document."""
    max_score: float = Field(..., ge=0.0, le=1.0, description="Maximum risk score for this label")
    high_risk_clauses: int = Field(..., ge=0, description="Number of high-risk clauses")
    total_clauses: int = Field(..., ge=0, description="Total clauses with this label")


class DocumentAnalysisResponse(BaseModel):
    """Complete document analysis response."""
    analysis_id: str
    document_risk: str = Field(..., description="Overall document risk level")
    doc_score: int = Field(..., description="Document risk score (average final_score * 10, rounded)")
    label_summary: Dict[str, LabelSummary] = Field(..., description="Per-label risk summaries")
    clauses: List[ClauseResult] = Field(..., description="Detailed clause-level analysis results")

    class Config:
        json_schema_extra = {
                "example": {
                    "analysis_id": '6c0dbc03-7f2b-4c85-a5cf-f10b34eef1eb',
                    "document_risk": "high_risk",
                    "doc_score": 7,
                    "label_summary": {
                        "non_compete": {
                            "max_score": 0.85,
                            "high_risk_clauses": 2,
                            "total_clauses": 5
                        }
                    },
                    "clauses": [
                        {
                            "page_no": 3,
                            "clause_text": "Employee agrees not to compete...",
                            "labels": [
                                {
                                    "label": "non_compete",
                                    "semantic_score": 0.82,
                                    "final_score": 0.78,
                                    "band": "high"
                                }
                            ],
                            "final_score": 0.78,
                            "identity": 0.95,
                            "semantic": 0.82,
                            "margin": 0.15,
                            # "top_matches": []
                        }
                    ]
                }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    ready: bool = Field(..., description="Whether service is ready to accept requests")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    message: str = Field(..., description="Status message")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")

