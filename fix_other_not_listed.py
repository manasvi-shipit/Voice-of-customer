"""
fix_other_not_listed.py

Applies the second-look relabeling of 'other_not_listed' rows discussed
between you and Claude -- read through RELABELS below BEFORE running.
If you disagree with any call, edit the value on that line first.

review_id -> new theme_1 (None means "leave as other_not_listed", i.e. keep it)
"""

import pandas as pd

GROUND_TRUTH_FILE = "ground_truth_final.csv"

# EDIT THESE if you disagree with any suggested relabel before running.
RELABELS = {
    "1e164bda-7d0a-4cba-9b41-cf23432ba4ea": "service_quality",
    "6eb27070-4d52-491b-b798-e380de7b5376": "over_priced",
    "73c53034-b1da-4d23-a40f-f9f8c9e77a60": "punctuality_noshow",
    "8ca5a19b-243b-42f0-8dcc-d3fc68987602": None,  # kept as other_not_listed - meta-complaint about reviews, not the service
    "cde84655-80a4-4b4d-acef-c4e928fab835": "booking_app_ux",
    "719b9a2e-f25e-42f2-8def-d62fcc8373e1": "booking_app_ux",
    "5c861a64-fb4c-402d-a93d-2810beaf12b5": "customer_support",
    "9ac0001b-124e-4bfa-ba5d-d5cbb95f95b1": "booking_app_ux",
    "77f0b288-5903-4752-b640-cac61d04ee31": "new_service_ask",
    "75ca3f1b-fa4a-4d34-9f9b-4e9dcdfbd51e": "booking_app_ux",
    "42f8be4d-ff67-41c2-bde7-e494d0822f28": "new_service_ask",
    "2b966e56-360c-4a3f-a4b9-6738315c1b42": "booking_app_ux",  # secondary theme customer_support -- add manually to theme_2_optional if you want
}


def main():
    df = pd.read_csv(GROUND_TRUTH_FILE)

    changed = 0
    for review_id, new_theme in RELABELS.items():
        if new_theme is None:
            continue
        mask = df["review_id"] == review_id
        if mask.sum() == 0:
            print(f"WARNING: review_id {review_id} not found in file -- skipped")
            continue
        df.loc[mask, "theme_1"] = new_theme
        changed += 1

    df.to_csv(GROUND_TRUTH_FILE, index=False)
    print(f"Updated {changed} rows.")
    print("\nNew theme_1 distribution:")
    print(df["theme_1"].value_counts())


if __name__ == "__main__":
    main()
