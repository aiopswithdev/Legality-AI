# main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import logging

from app.models import initialize_models
from app.pipeline import analyze_document

# -------------------------------------------------
# Logging configuration
# -------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("api")


# -------------------------------------------------
# Lifespan handler (replaces on_event)
# -------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")
    initialize_models()          # 🔥 load models ONCE
    logger.info("Application startup complete.")
    yield
    logger.info("Shutting down application...")


# -------------------------------------------------
# FastAPI app
# -------------------------------------------------

app = FastAPI(
    title="Legal Clause Risk Detector",
    version="1.0.0",
    lifespan=lifespan
)


# -------------------------------------------------
# API endpoint
# -------------------------------------------------

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    return analyze_document(pdf_bytes)