import json
import time
from pathlib import Path

from langfuse import Langfuse
from app.call_llm_gemini import call_llm_gemini
from app.judge_eval.utils import clause_fingerprint
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict

load_dotenv()

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

DATASET_PATH = Path("data/judge_dataset_v1.jsonl")
SKIP_FILE_PATH = Path("data/processed_clauses.txt") # The "Skip Database"
ALLOWED_LABELS = {
    "uncapped_liability",
    "termination_for_convenience",
}
def load_skip_ids(file_path: Path) -> set:
    """Loads processed IDs from a file into a set."""
    if not file_path.exists():
        return set()
    with file_path.open("r") as f:
        # strip() removes newlines, filter removes empty lines
        return set(line.strip() for line in f if line.strip())

def save_skip_id(file_path: Path, clause_id: str):
    """Appends a single ID to the skip file."""
    with file_path.open("a") as f:
        f.write(f"{clause_id}\n")
DEFAULT_SLEEP = 2.0

SKIP_CLAUSE_ID = {
    "a99777ec56216637370fdaf6c4a933d29544e527c987cc45f2b6e6286cd9bea9",
    "cec5f3cd1cbe3d0449d13259c658bfddae88ecc2a226160ca7259f909c573698",
    "ff9bfdb1de2d8b724e1b625a854ef9370a9ea373e2c58314929b22005db4fdd2",
    "27c3c9a10aa2bd3a8832e46f409e6f0cde80d4560b161bb15117ce654cbfd222"
}

def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    
    # 1. Load previously processed IDs
    processed_ids = load_skip_ids(SKIP_FILE_PATH)
    print(f"Loaded {len(processed_ids)} IDs to skip.")

    langfuse = Langfuse()

    total_rows_seen = 0
    total_judged = 0
    total_failed = 0

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total_rows_seen += 1
            clause_id = row.get('clause_id')

            # ---- Scope control ----
            if row.get("band") != "review":
                continue

            if row.get("label") not in ALLOWED_LABELS:
                continue

            # 2. Check if already processed
            if clause_id in processed_ids:
                continue

            # ---- Create event FIRST ----
            event = langfuse.create_event(
                name="llm_judge",
                input={
                    "clause_text": row["clause_text"],
                    "label": row["label"],
                    "model_band": row["band"],
                },
                metadata={
                    "analysis_id": row["analysis_id"],
                },
                version="gemini-judge-v1",
            )

            # ---- Call Gemini judge (single clause mode) ----
            try:
                llm_result = call_llm_gemini(
                    clause_text=row["clause_text"],
                    label=row["label"],
                )
            except Exception as e:
                total_failed += 1
                print("⚠️ Gemini error on selected clause")
                print(e)
                break  # intentional: stop after first failure

            # ---- Log LLM judgments as SCORES ----
            langfuse.create_score(
                trace_id=event.trace_id,
                name="llm_band",
                value=llm_result["suggested_band"],
            )

            langfuse.create_score(
                trace_id=event.trace_id,
                name="llm_relevance",
                value=str(llm_result["relevant"]),
            )

            langfuse.create_score(
                trace_id=event.trace_id,
                name="llm_confidence",
                value=float(llm_result["confidence"]),
            )

            total_judged += 1
            save_skip_id(SKIP_FILE_PATH, clause_id)
            break  # stop after ONE successful judgment

    langfuse.flush()

    print("✅ LLM judge run complete (single clause mode)")
    print(f"📄 Total rows scanned: {total_rows_seen}")
    print(f"🤖 Clauses judged: {total_judged}")
    print(f"❌ Gemini failures: {total_failed}")


if __name__ == "__main__":
    main()
