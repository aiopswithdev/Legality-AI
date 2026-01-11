import json
from typing import List
from app.config import RISK_THRESHOLDS
from app.judge_eval.schemas import JudgeInput
from app.judge_eval.sampler import should_sample
from app.judge_eval.utils import hash_clause


def build_judge_dataset(
    analysis_results: List[dict],
    threshold_version: str = "v1.0"
) -> List[JudgeInput]:

    rows = []

    for result in analysis_results:
        analysis_id = result["analysis_id"]

        for clause in result["clauses"]:
            clause_text = clause["clause_text"]

            for label_info in clause["labels"]:
                label = label_info["label"]

                low_thr, high_thr = RISK_THRESHOLDS.get(
                    label,
                    (0.60, 0.70)
                )

                # inject thresholds so sampler can see them
                label_info["threshold_low"] = low_thr

                if not should_sample(label_info, clause):
                    continue

                rows.append(
                    JudgeInput(
                        clause_id=hash_clause(clause_text, label),
                        analysis_id=analysis_id,

                        clause_text=clause_text,
                        label=label,

                        identity=clause["identity"],
                        semantic_score=label_info["semantic_score"],
                        margin=clause["margin"],
                        final_score=label_info["final_score"],
                        band=label_info["band"],

                        threshold_low=low_thr,
                        threshold_high=high_thr,

                        threshold_version=threshold_version
                    )
                )

    return rows
