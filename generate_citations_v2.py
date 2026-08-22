"""
generate_citations_v2.py

Stage 2 of the Voice of Customer Copilot agent pipeline (v2 -- paired
with classify_reviews_v2.py's universal taxonomy / any-company mode).

For each theme, asks the LLM to pick which review_ids best represent it --
but the LLM NEVER generates quote text itself. The actual verbatim quote
is always fetched by code from the original source CSV. This makes
quote fabrication structurally impossible, rather than relying on
after-the-fact verification.

This script itself needed no logic changes for the universal-taxonomy /
multi-company generalization -- it was already domain-agnostic (theme
names are read dynamically from classified_reviews_v2.csv, nothing
taxonomy-specific is hardcoded here). Only the input/output filenames
changed, to keep this run separate from the validated Urban Company v1
output (theme_report.csv).

GUARDRAILS in this stage (unchanged from v1):
1. Quotes are always the real source text, looked up by ID -- never
   LLM-generated text.
2. Any review_id the model returns that isn't actually in that theme's
   review set is dropped and logged (defends against a hallucinated ID).
3. Themes with fewer than MIN_REVIEWS_FOR_CITATION supporting reviews are
   marked LOW_CONFIDENCE and excluded from the "reliable" theme list
   rather than forcing a citation onto thin evidence (PRD requirement F6).

SETUP: same as classify_reviews.py (google-genai package, GEMINI_API_KEY env var)

INPUT:  urban_company_reviews.csv, classified_reviews_v2.csv
OUTPUT: theme_report_v2.csv (theme, review_count, confidence, summary, cited quotes)
"""

import os
import csv
import json
import time
import re
from collections import defaultdict

from google import genai

REVIEWS_FILE = "custom_reviews.csv"  # dedicated file for any-company mode --
                                       # NEVER urban_company_reviews.csv (see
                                       # classify_reviews_v2.py for why)
CLASSIFICATIONS_FILE = "classified_reviews_v2.csv"
OUTPUT_FILE = "theme_report_v2.csv"

MODEL_NAME = "gemini-3.5-flash-lite"
MIN_REVIEWS_FOR_CITATION = 3  # PRD requirement F6 -- below this, a theme
                                # is flagged low-confidence rather than
                                # forced into the report
QUOTES_PER_THEME = 3

MAX_RETRIES = 4
BASE_RETRY_DELAY_SECONDS = 20
SECONDS_BETWEEN_CALLS = 15  # same 5 RPM account pacing as Stage 1


def load_and_merge():
    with open(REVIEWS_FILE, encoding="utf-8") as f:
        reviews = {r["review_id"]: r for r in csv.DictReader(f)}

    with open(CLASSIFICATIONS_FILE, encoding="utf-8") as f:
        classifications = list(csv.DictReader(f))

    # Group review_ids by theme_1. (theme_2_optional is intentionally
    # excluded from grouping here -- a documented simplification. Secondary
    # themes are rarer and adding them would double-count some reviews
    # across themes in the frequency counts used for prioritization later.)
    theme_groups = defaultdict(list)
    for c in classifications:
        theme = (c.get("theme_1") or "").strip()
        review_id = c.get("review_id")
        if theme and review_id in reviews:
            theme_groups[theme].append({
                "review_id": review_id,
                "rating": reviews[review_id]["rating"],
                "review_text": reviews[review_id]["review_text"],
            })
        elif theme and review_id not in reviews:
            print(f"  WARNING: review_id {review_id} in classifications but not in source reviews -- skipped")

    return theme_groups


def build_prompt(theme, theme_reviews):
    reviews_lines = "\n".join(
        f'{{"review_id": "{r["review_id"]}", "rating": {r["rating"]}, "text": "{r["review_text"].replace(chr(34), chr(39))}"}}'
        for r in theme_reviews
    )

    prompt = f"""You are selecting the most representative reviews for the customer-feedback theme "{theme}".

Below are {len(theme_reviews)} reviews already classified under this theme:
{reviews_lines}

Select up to {QUOTES_PER_THEME} review_ids that best represent this theme -- prioritize reviews that are specific, information-rich, and cover different aspects of the theme rather than repeating the same point. Do NOT select near-duplicate reviews.

Also write a one-paragraph summary (2-3 sentences) describing the pattern this theme captures, based on what you see across all {len(theme_reviews)} reviews.

Return ONLY valid JSON, no markdown formatting, in this exact structure:
{{"selected_review_ids": ["id1", "id2", "id3"], "summary": "..."}}

Every id in selected_review_ids MUST be one of the review_ids given above -- do not invent or guess an id."""
    return prompt


def extract_json(raw_text):
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def call_model_for_theme(client, theme, theme_reviews):
    prompt = build_prompt(theme, theme_reviews)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            raw = extract_json(response.text)
            parsed = json.loads(raw)
            return parsed
        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                wait_time = BASE_RETRY_DELAY_SECONDS * attempt
                print(f"  Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
            else:
                print(f"  Giving up on theme '{theme}' after {MAX_RETRIES} attempts.")
                return None


def process_theme(client, theme, theme_reviews):
    review_count = len(theme_reviews)

    if review_count < MIN_REVIEWS_FOR_CITATION:
        return {
            "theme": theme,
            "review_count": review_count,
            "confidence": "LOW_CONFIDENCE_NEEDS_HUMAN_REVIEW",
            "summary": f"Only {review_count} review(s) support this theme -- below the minimum of {MIN_REVIEWS_FOR_CITATION} required for a reliable citation. Needs manual review rather than automated summarization.",
            "cited_review_ids": "",
            "cited_quotes": "",
        }

    result = call_model_for_theme(client, theme, theme_reviews)

    valid_ids_for_theme = {r["review_id"] for r in theme_reviews}
    id_to_text = {r["review_id"]: r["review_text"] for r in theme_reviews}

    if result is None:
        return {
            "theme": theme,
            "review_count": review_count,
            "confidence": "MODEL_CALL_FAILED_NEEDS_HUMAN_REVIEW",
            "summary": "",
            "cited_review_ids": "",
            "cited_quotes": "",
        }

    selected_ids = result.get("selected_review_ids", [])

    # GUARDRAIL: drop any id the model returned that isn't actually in
    # this theme's real review set -- a hallucinated or wrong id never
    # makes it into the output.
    verified_ids = []
    for rid in selected_ids:
        if rid in valid_ids_for_theme:
            verified_ids.append(rid)
        else:
            print(f"  GUARDRAIL: model returned review_id '{rid}' which is not in theme '{theme}' -- dropped")

    # The actual quote text is ALWAYS fetched from source data by code,
    # never taken from the model's output.
    verified_quotes = [id_to_text[rid] for rid in verified_ids]

    return {
        "theme": theme,
        "review_count": review_count,
        "confidence": "OK",
        "summary": result.get("summary", ""),
        "cited_review_ids": "; ".join(verified_ids),
        "cited_quotes": " || ".join(verified_quotes),
    }


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Run: export GEMINI_API_KEY='your-key'")

    client = genai.Client(api_key=api_key)

    theme_groups = load_and_merge()
    print(f"Found {len(theme_groups)} distinct themes across all classified reviews.\n")

    reports = []
    for theme, theme_reviews in sorted(theme_groups.items(), key=lambda x: -len(x[1])):
        print(f"Processing theme '{theme}' ({len(theme_reviews)} reviews)...")
        report = process_theme(client, theme, theme_reviews)
        reports.append(report)

        if report["confidence"] == "OK":
            time.sleep(SECONDS_BETWEEN_CALLS)  # only pause if we actually made an API call

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["theme", "review_count", "confidence", "summary", "cited_review_ids", "cited_quotes"])
        writer.writeheader()
        for r in reports:
            writer.writerow(r)

    print(f"\nDone. Wrote {len(reports)} theme reports to {OUTPUT_FILE}")
    low_confidence = [r for r in reports if r["confidence"] != "OK"]
    if low_confidence:
        print(f"{len(low_confidence)} theme(s) flagged as low-confidence/failed -- see the report for details.")


if __name__ == "__main__":
    main()
