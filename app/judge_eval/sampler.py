def should_sample(label_info: dict, clause: dict) -> bool:
    band = label_info["band"]
    final_score = label_info["final_score"]
    margin = clause["margin"]

    # Always sample REVIEW
    if band == "review":
        return True

    # High risk but low confidence
    if band == "high" and margin < 0.05:
        return True

    # Near LOW threshold (decision boundary)
    threshold_low = label_info.get("threshold_low", 0.60)
    if abs(final_score - threshold_low) < 0.05:
        return True

    return False
