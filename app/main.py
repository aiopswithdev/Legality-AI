"""
FastAPI application for Legal Clause Risk Detection.
Production-ready with CORS, health checks, error handling, and validation.
"""
from app.analysis_cache import create_analysis_entry, get_analysis_entry
from app.pdf_highlight import highlight_clauses_in_pdf
from fastapi.responses import StreamingResponse
import io

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import os
from typing import Dict, Any

from app import models #initialize_models, embed_model, reranker, faiss_index, metadata
from app.pipeline import analyze_document
from app.schemas import (
    DocumentAnalysisResponse,
    HealthResponse,
    ReadinessResponse,
    ErrorResponse
)
from app.exceptions import (
    LegalityAIException,
    ModelNotLoadedError,
    InvalidFileError,
    FileProcessingError
)
from app.config import (
    CORS_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_HEADERS,
    MAX_FILE_SIZE_BYTES,
    LOG_LEVEL
)

# -------------------------------------------------
# Logging configuration
# -------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("api")
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))


# -------------------------------------------------
# Lifespan handler
# -------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logger.info("=" * 60)
    logger.info("Starting Legality-AI Application...")
    logger.info("=" * 60)
    
    try:
        models.initialize_models()
        logger.info("✅ Application startup complete. All models loaded.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}", exc_info=True)
        raise
    
    yield
    
    logger.info("=" * 60)
    logger.info("Shutting down application...")
    logger.info("=" * 60)


# -------------------------------------------------
# FastAPI app
# -------------------------------------------------

app = FastAPI(
    title="Legal Clause Risk Detector API",
    description="AI-powered API for analyzing legal documents and detecting risky clauses",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# -------------------------------------------------
# CORS Middleware
# -------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)


# -------------------------------------------------
# Request Logging Middleware
# -------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing information."""
    start_time = time.time()
    
    # Log request
    logger.info(f"→ {request.method} {request.url.path} - Client: {request.client.host if request.client else 'unknown'}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"← {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"✗ {request.method} {request.url.path} - "
            f"Error: {str(e)} - "
            f"Time: {process_time:.3f}s",
            exc_info=True
        )
        raise


# -------------------------------------------------
# Exception Handlers
# -------------------------------------------------

@app.exception_handler(LegalityAIException)
async def legality_ai_exception_handler(request: Request, exc: LegalityAIException):
    """Handle custom application exceptions."""
    logger.error(f"Application error: {exc}", exc_info=True)
    
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = "InternalServerError"
    
    if isinstance(exc, ModelNotLoadedError):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        error_type = "ServiceUnavailable"
    elif isinstance(exc, InvalidFileError):
        status_code = status.HTTP_400_BAD_REQUEST
        error_type = "InvalidFile"
    elif isinstance(exc, FileProcessingError):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        error_type = "FileProcessingError"
    
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=error_type,
            message=str(exc),
            detail=None
        ).model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            detail=None
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail=str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None
        ).model_dump()
    )


# -------------------------------------------------
# Health Check Endpoints
# -------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Basic health check endpoint to verify the API is running."
)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["Health"],
    summary="Readiness Check",
    description="Check if the service is ready to accept requests (models loaded)."
)
async def readiness_check():
    """Readiness check - verifies models are loaded."""
    models_loaded = all([
        models.embed_model is not None,
        models.reranker is not None,
        models.faiss_index is not None,
        models.metadata is not None and len(models.metadata) > 0
    ])
    
    if models_loaded:
        return ReadinessResponse(
            ready=True,
            models_loaded=True,
            message="Service is ready"
        )
    else:
        return ReadinessResponse(
            ready=False,
            models_loaded=False,
            message="Models not loaded"
        )


# -------------------------------------------------
# Main API Endpoint
# -------------------------------------------------

@app.post(
    "/analyze",
    response_model=DocumentAnalysisResponse,
    tags=["Analysis"],
    summary="Analyze Document",
    description="Upload a PDF document to analyze for risky legal clauses.",
    responses={
        200: {
            "description": "Analysis completed successfully",
            "model": DocumentAnalysisResponse
        },
        400: {
            "description": "Invalid file or request",
            "model": ErrorResponse
        },
        413: {
            "description": "File too large",
            "model": ErrorResponse
        },
        422: {
            "description": "File processing error",
            "model": ErrorResponse
        },
        503: {
            "description": "Service unavailable (models not loaded)",
            "model": ErrorResponse
        }
    }
)
async def analyze(
    request: Request,
    file: UploadFile = File(
        ...,
        description="PDF file to analyze"
        # max_length=255
    )
):
    """
    Analyze a PDF document for risky legal clauses.
    
    - **file**: PDF file to analyze (max size: 50MB by default)
    
    Returns a comprehensive analysis including:
    - Document-level risk assessment
    - Per-label risk summaries
    - Detailed clause-level analysis with risk scores
    """
    # Check if models are loaded
    if not all([models.embed_model, models.reranker, models.faiss_index, models.metadata]):
        raise ModelNotLoadedError("ML models are not loaded. Service is not ready.")
    
    # Validate file extension
    if not file.filename:
        raise InvalidFileError("Filename is required")
    
    if not file.filename.lower().endswith(".pdf"):
        raise InvalidFileError("Only PDF files are supported")

    # Read file with size validation
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise InvalidFileError(f"Failed to read file: {str(e)}")
    
    # Validate file size
    file_size = len(pdf_bytes)
    if file_size == 0:
        raise InvalidFileError("Empty file")
    
    if file_size > MAX_FILE_SIZE_BYTES:
        max_size_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {max_size_mb:.0f}MB"
        )
    
    logger.info(
        f"Processing file: {file.filename} "
        f"(Size: {file_size / (1024 * 1024):.2f}MB)"
    )
    
    # Process document
    try:
        result = analyze_document(pdf_bytes)
        logger.info(
            f"Analysis complete for {file.filename}. "
            f"Found {len(result.get('clauses', []))} risky clauses."
        )

        analysis_id = create_analysis_entry(
            pdf_bytes=pdf_bytes,
            analysis_result=result
        )

        # return analysis_id + original response (includes doc_score)
        return {
            "analysis_id": analysis_id,
            **result
        }
    
    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        raise FileProcessingError(f"Failed to process document: {str(e)}")

@app.get(
    "/highlight/{analysis_id}",
    tags=["Analysis"],
    summary="Download highlighted PDF from cached analysis"
)
async def download_highlighted_pdf(analysis_id: str):
    entry = get_analysis_entry(analysis_id)

    if not entry:
        raise HTTPException(
            status_code=404,
            detail="Analysis expired or not found. Please re-analyze."
        )

    highlighted_pdf = highlight_clauses_in_pdf(
        pdf_bytes=entry["pdf_bytes"],
        clauses=entry["result"].get("clauses", [])
    )

    return StreamingResponse(
        io.BytesIO(highlighted_pdf),
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=highlighted_contract.pdf"
        }
    )

# -------------------------------------------------
# Root endpoint
# -------------------------------------------------

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Legal Clause Risk Detector API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready"
    }
