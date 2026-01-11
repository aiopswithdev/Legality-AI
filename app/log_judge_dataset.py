"""
Logs frozen judge dataset to Langfuse as `judge_input` traces.

Assumes:
- data/judge_dataset_v1.jsonl exists
- Langfuse keys are set in env
"""

import json
from pathlib import Path
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()
DATASET_PATH = Path("data/judge_dataset_v1.jsonl")


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    langfuse = Langfuse()  # keys picked from env

    total = 0
    try: 
        with DATASET_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                # ---- INPUT (what will later be judged) ----
                input_payload = {
                    "clause_text": row["clause_text"],
                    "final_score": row["final_score"],
                    "identity": row["identity"],
                    "semantic_score": row["semantic_score"],
                    "margin": row["margin"],
                }

                # ---- METADATA (observability only) ----
                metadata_payload = {
                    "clause_id": row["clause_id"],
                    "analysis_id": row["analysis_id"],
                    "threshold_low": row["threshold_low"],
                    "threshold_high": row["threshold_high"],
                }
                event = langfuse.create_event(
                    # attributes=attributes,
                    name="judge_input",
                    input=input_payload,
                    metadata=metadata_payload,
                    version=row["threshold_version"],
                )

                # 2️⃣ Attach INDEXABLE scores (THIS IS THE KEY)
                langfuse.create_score(
                    trace_id=event.trace_id,
                    name="band",
                    value=row["band"],
                )

                langfuse.create_score(
                    trace_id=event.trace_id,
                    name="label",
                    value=row["label"],
                )
                langfuse.create_score(
                trace_id=event.trace_id,
                name="label_in_band",
                value=f"{row['band']}::{row['label']}"
            )
                total += 1
    finally:
        # CRITICAL: Ensures all data is sent before script exits
        langfuse.flush()
    print("✅ Judge dataset logged to Langfuse")
    print(f"📊 Total events created: {total}")


if __name__ == "__main__":
    main()
