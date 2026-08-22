"""
pull_urban_company_reviews.py

Pulls Urban Company (customer app) reviews from the Google Play Store
and saves them as a clean CSV for the Voice of Customer Copilot project.

Run this on your own machine (not in a sandboxed/restricted-network environment) —
it needs live internet access to reach play.google.com.
"""

import csv
import time
from google_play_scraper import reviews, Sort

# Urban Company's customer-facing app package name on Play Store
# (confirmed via play.google.com/store/apps/details?id=com.urbanclap.urbanclap)
APP_ID = "com.urbanclap.urbanclap"

# How many reviews to pull. 500 is a good MVP volume — enough for real
# theme diversity without making manual labeling (50-80 of these) unwieldy.
TARGET_COUNT = 500

# Pull newest first — most relevant to current product issues.
SORT_ORDER = Sort.NEWEST

OUTPUT_FILE = "urban_company_reviews.csv"


def pull_reviews(app_id: str, target_count: int, sort_order):
    """
    Paginates through the Play Store reviews API until target_count is reached
    or no more reviews are available. The API returns reviews in batches with
    a continuation token, so we loop until we have enough.
    """
    all_reviews = []
    continuation_token = None
    batch_size = 200  # max per request the library handles well

    while len(all_reviews) < target_count:
        result, continuation_token = reviews(
            app_id,
            lang="en",       # English-language reviews only, for consistent NLP processing
            country="in",    # India store — matches Urban Company's primary market
            sort=sort_order,
            count=batch_size,
            continuation_token=continuation_token,
        )

        if not result:
            print("No more reviews returned by the API — stopping early.")
            break

        all_reviews.extend(result)
        print(f"Pulled {len(all_reviews)} reviews so far...")

        if continuation_token is None:
            # No more pages available
            break

        time.sleep(1)  # be polite to the API, avoid rate-limiting

    return all_reviews[:target_count]


def clean_and_save(raw_reviews, output_file: str):
    """
    Extracts only the fields we need for the pipeline and writes a clean CSV:
    review_id, review_text, rating, date, thumbs_up_count
    """
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review_id", "review_text", "rating", "date", "thumbs_up_count"])

        for r in raw_reviews:
            review_id = r.get("reviewId", "")
            review_text = (r.get("content") or "").replace("\n", " ").strip()
            rating = r.get("score", "")
            date = r.get("at", "")
            thumbs_up = r.get("thumbsUpCount", 0)

            # Skip empty or extremely short reviews — not useful for theme clustering
            if len(review_text) < 15:
                continue

            writer.writerow([review_id, review_text, rating, date, thumbs_up])

    print(f"Saved cleaned reviews to {output_file}")


if __name__ == "__main__":
    print(f"Pulling up to {TARGET_COUNT} reviews for app '{APP_ID}'...")
    raw = pull_reviews(APP_ID, TARGET_COUNT, SORT_ORDER)
    print(f"Total raw reviews pulled: {len(raw)}")

    clean_and_save(raw, OUTPUT_FILE)
    print("Done. Open the CSV to eyeball the data before building the pipeline.")
