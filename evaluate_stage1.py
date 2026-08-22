"""
evaluate_stage1.py

Compares the agent's Stage 1 classifications (classified_reviews.csv)
against your hand-labeled ground truth (ground_truth_final.csv) for the
92 reviews you personally labeled.

This is the actual eval -- the number that matters for the portfolio.
A realistic first-pass score is often 50-70%, not 90%+. Document whatever
comes back honestly; the iteration story (what broke, what you fixed) is
more valuable than a suspiciously perfect first run.

INPUT:  ground_truth_final.csv, classified_reviews.csv
OUTPUT: eval_report.txt (printed metrics), eval_mismatches.csv (rows where
        the agent disagreed with you, for manual inspection)
"""

import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support

GROUND_TRUTH_FILE = "ground_truth_final.csv"
PREDICTIONS_FILE = "classified_reviews.csv"
MISMATCH_OUTPUT_FILE = "eval_mismatches.csv"


def normalize(val):
    """Lowercase + strip, so minor casing/whitespace differences don't
    cause false mismatches in the comparison."""
    if pd.isna(val):
        return ""
    return str(val).strip().lower()


def main():
    gt = pd.read_csv(GROUND_TRUTH_FILE)
    pred = pd.read_csv(PREDICTIONS_FILE)

    # Merge on review_id -- inner join keeps only the 92 reviews you
    # actually hand-labeled, since that's the only set we have a real
    # answer key for.
    merged = gt.merge(pred, on="review_id", suffixes=("_human", "_agent"))
    print(f"Matched {len(merged)} of {len(gt)} ground-truth rows to agent predictions.\n")

    if len(merged) < len(gt):
        missing = set(gt["review_id"]) - set(pred["review_id"])
        print(f"WARNING: {len(missing)} ground-truth review_ids were not found in the agent's output.")
        print("This could mean the agent's MISSING_FROM_LLM_RESPONSE guardrail caught them, or something else dropped them.\n")

    # --- Theme_1 exact match accuracy ---
    merged["theme_1_human_norm"] = merged["theme_1_human"].apply(normalize)
    merged["theme_1_agent_norm"] = merged["theme_1_agent"].apply(normalize)
    merged["theme_1_exact_match"] = merged["theme_1_human_norm"] == merged["theme_1_agent_norm"]

    exact_accuracy = merged["theme_1_exact_match"].mean()
    print(f"=== THEME_1 EXACT MATCH ACCURACY ===")
    print(f"{exact_accuracy:.1%} ({merged['theme_1_exact_match'].sum()} of {len(merged)} correct)\n")

    # --- Per-category precision / recall / F1 ---
    print("=== PER-CATEGORY PRECISION / RECALL / F1 (theme_1) ===")
    print(classification_report(
        merged["theme_1_human_norm"],
        merged["theme_1_agent_norm"],
        zero_division=0,
    ))

    # --- Churn risk accuracy ---
    merged["churn_human_norm"] = merged["is_churn_risk_human"].apply(normalize)
    merged["churn_agent_norm"] = merged["is_churn_risk_agent"].apply(normalize)
    merged["churn_match"] = merged["churn_human_norm"] == merged["churn_agent_norm"]

    churn_accuracy = merged["churn_match"].mean()
    print(f"=== CHURN RISK FLAG ACCURACY ===")
    print(f"{churn_accuracy:.1%} ({merged['churn_match'].sum()} of {len(merged)} correct)\n")

    precision, recall, f1, _ = precision_recall_fscore_support(
        merged["churn_human_norm"],
        merged["churn_agent_norm"],
        pos_label="y",
        average="binary",
        zero_division=0,
    )
    print(f"Churn risk (Y) -- Precision: {precision:.1%}, Recall: {recall:.1%}, F1: {f1:.1%}\n")

    # --- Save mismatches for manual inspection ---
    mismatches = merged[~merged["theme_1_exact_match"] | ~merged["churn_match"]][[
        "review_id", "review_text", "rating",
        "theme_1_human", "theme_1_agent",
        "is_churn_risk_human", "is_churn_risk_agent",
    ]]
    mismatches.to_csv(MISMATCH_OUTPUT_FILE, index=False)
    print(f"Saved {len(mismatches)} mismatched rows to {MISMATCH_OUTPUT_FILE} for manual review.")


if __name__ == "__main__":
    main()
