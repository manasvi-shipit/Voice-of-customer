"""
create_labeling_sample.py

Builds a stratified sample from urban_company_reviews.csv for manual
ground-truth labeling. Pulls ALL reviews from the rare middle-rating
buckets (2, 3, 4 star) and a random sample from the extremes (1 and 5
star), so the labeled set actually reflects the full range of feedback
rather than being dominated by whichever rating has the most volume.

Run this in the same folder as urban_company_reviews.csv.
"""

import csv
import random

random.seed(42)  # reproducible sample

INPUT_FILE = "urban_company_reviews.csv"
OUTPUT_FILE = "labeling_sheet.csv"

# How many to sample from each rating bucket.
# 2, 3, 4 star are rare in this dataset — take all of them.
# 1 and 5 star are abundant — cap them so the label set isn't
# dominated by just the extremes.
SAMPLE_CAPS = {
    "1": 20,
    "2": None,  # None = take all available
    "3": None,
    "4": None,
    "5": 20,
}

# Starter theme taxonomy — based on the PRD's initial hypothesis.
# You WILL adjust this once you start reading real reviews; treat
# this as a first draft, not gospel.
THEME_OPTIONS = [
    "punctuality_noshow",
    "service_quality",
    "pricing_transparency",
    "refund_cancellation",
    "booking_app_ux",
    "safety_trust",
    "customer_support",
    "other_not_listed",
]


def load_reviews(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stratified_sample(rows):
    by_rating = {}
    for row in rows:
        by_rating.setdefault(row["rating"], []).append(row)

    sample = []
    for rating, cap in SAMPLE_CAPS.items():
        bucket = by_rating.get(rating, [])
        if cap is None or len(bucket) <= cap:
            sample.extend(bucket)
        else:
            sample.extend(random.sample(bucket, cap))

    random.shuffle(sample)  # so you're not labeling in rating-sorted blocks
    return sample


def write_labeling_sheet(sample, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "review_id",
            "review_text",
            "rating",
            "date",
            "theme_1",           # fill in manually
            "theme_2_optional",  # fill in manually, leave blank if only one theme applies
            "is_churn_risk",     # fill in manually: Y or N
            "your_summary_note", # fill in manually: 1 short sentence, what would a good AI summary say
        ])
        for row in sample:
            writer.writerow([
                row["review_id"],
                row["review_text"],
                row["rating"],
                row["date"],
                "", "", "", "",
            ])
    print(f"Wrote {len(sample)} rows to {path}")
    print(f"\nTheme options to use in theme_1 / theme_2_optional columns:")
    for t in THEME_OPTIONS:
        print(f"  - {t}")
    print("\nIf a review doesn't fit any of these, use 'other_not_listed' and")
    print("note in your_summary_note what the actual theme seems to be —")
    print("that's exactly the signal that tells you the taxonomy needs updating.")


if __name__ == "__main__":
    rows = load_reviews(INPUT_FILE)
    sample = stratified_sample(rows)
    write_labeling_sheet(sample, OUTPUT_FILE)
