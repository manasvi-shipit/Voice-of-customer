"""
classify_reviews.py

Stage 1 of the Voice of Customer Copilot agent pipeline.

Classifies each Urban Company review against the FIXED taxonomy you built
during manual ground-truth labeling. Classifying against a fixed taxonomy
(rather than open-ended clustering) means every prediction can be directly
compared to your human labels for precision/recall -- that comparison is
what turns this into a measurable eval, not just "the output looks fine."

SETUP:
    pip install google-genai

    (Note: the older `google-generativeai` package is deprecated as of
    late 2025 -- this script uses the current `google-genai` package.)

    Set your API key as an environment variable (never hardcode it in the
    script -- this is also a small guardrail/security practice worth
    mentioning in interviews):

    export GEMINI_API_KEY="your-key-here"

USAGE:
    python classify_reviews.py

INPUT:  urban_company_reviews.csv  (review_id, review_text, rating, date, thumbs_up_count)
OUTPUT: classified_reviews.csv     (review_id, theme_1, theme_2_optional, is_churn_risk)
"""

import os
import csv
import json
import time
import re

from google import genai

INPUT_FILE = "urban_company_reviews.csv"
OUTPUT_FILE = "classified_reviews.csv"

BATCH_SIZE = 40   # reviews per API call -- small enough to keep output
                   # JSON reliable, large enough to be efficient
MAX_RETRIES = 4
BASE_RETRY_DELAY_SECONDS = 20  # your account's free-tier limit is 5
                                 # requests/minute (confirmed from a live
                                 # 429 error), so retries need real spacing,
                                 # not a quick 1-2 second bounce
SECONDS_BETWEEN_BATCHES = 15    # ~4 requests/minute, safely under the 5 RPM cap

MODEL_NAME = "gemini-3.5-flash-lite"  # gemini-2.5-flash and the entire 2.x
                                        # line are blocked for new accounts as
                                        # of Aug 2026 (a known, currently open
                                        # Google issue -- the model still
                                        # appears in the models.list() response
                                        # but 404s on the actual generate call).
                                        # 3.5-flash-lite is confirmed working
                                        # for new accounts.

# Known aliases for categories that were merged/renamed. If the model
# outputs one of these (because it's semantically sensible to it, even
# though the prompt no longer lists it as valid), we remap rather than
# silently accepting an out-of-taxonomy value.
THEME_ALIASES = {
    "pricing_transparency": "over_priced",  # merged during Stage 1 eval iteration
}

# The exact taxonomy from your ground truth labeling. Kept as the literal
# strings you used (including the ones with spaces/apostrophes) so eval
# comparisons later are a clean, direct string match against
# ground_truth_final.csv -- no risk of the model inventing a slightly
# different name for the same category.
TAXONOMY = {
    "punctuality_noshow": "Professional arriving late, not showing up, or no communication about delays",
    "service_quality": "Quality of the actual work performed -- skill, redo requests, workmanship",
    "over_priced": "ANY pricing complaint -- general cost complaints AND specific quoted-vs-charged mismatches. (Note: this category was previously split into 'over_priced' and 'pricing_transparency', but the boundary proved impossible to apply consistently in eval testing -- merged into one category.)",
    "refund_cancellation": "Refund delays/denials or cancellation friction",
    "booking_app_ux": "Issues with the app itself -- slot availability, rescheduling, booking flow",
    "safety_trust": "Safety concerns, inappropriate behavior, trust/verification issues",
    "customer_support": "Poor or unresponsive customer support experience",
    "generic_praise": "Positive review with NO specific actionable detail (e.g. 'excellent service', 'good work')",
    "competitor's_praise": "Review explicitly compares Urban Company unfavorably to a named competitor",
    "new_service_ask": "Review requests a new feature or service capability that doesn't exist yet",
    "loyalty program ask": "Review asks for discounts/offers for repeat/loyal customers",
    "useless review": "Review contains no real feedback signal at all -- not even generic praise (e.g. a stray request unrelated to feedback)",
    "other_not_listed": "Doesn't fit any category above -- use sparingly, and only when genuinely nothing else fits",
}


def build_prompt(batch):
    taxonomy_lines = "\n".join(f'- "{name}": {desc}' for name, desc in TAXONOMY.items())
    reviews_lines = "\n".join(
        f'{{"review_id": "{r["review_id"]}", "rating": {r["rating"]}, "text": "{r["review_text"].replace(chr(34), chr(39))}"}}'
        for r in batch
    )

    prompt = f"""You are classifying customer reviews for Urban Company (a home services app) into a FIXED taxonomy. Do not invent new categories -- pick the closest fit from the list below, even if imperfect.

TAXONOMY (theme name: description):
{taxonomy_lines}

For each review below, assign:
- theme_1: the single best-fitting theme name from the taxonomy above (exact string match required)
- theme_2_optional: a second theme ONLY if the review clearly covers two distinct themes, otherwise null
- is_churn_risk: "Y" if the review signals real dissatisfaction that puts continued use of the app at risk -- this includes explicit statements of leaving/switching, but also strong statements like "never recommend," repeated severe operational failures described with frustration, or loss of trust in the platform. Simple negative adjectives ("bad," "worst," "pathetic") on their own, with no further context, are NOT sufficient -- but do not require an exact phrase match to the examples below; generalize the underlying signal.

  Examples of Y (real signal that the customer may stop using the platform):
  - "Not better than Snabbit, switching to them from now on" -> Y (named competitor comparison)
  - "Uninstalling this app, never booking again" -> Y (explicit stated action)
  - "I wanted to cancel my existing plan" -> Y (active cancellation attempt)
  - "Never Recommend Again to Anyone" -> Y (strong recommendation-withdrawal statement)
  - "3 bookings cancelled, 2 payment failures, wasted so much of my time" -> Y (repeated severe operational failure + explicit frustration, even with no explicit "switching" language)
  - "Lost my trust" -> Y (explicit statement of broken trust in the platform)

  Examples of N (negative/angry but NOT a real signal of leaving):
  - "Worst service ever, rating must be fake" -> N (harsh language, but no further signal beyond the insult itself)
  - "Rates are disguised, hidden charges everywhere" -> N (pricing complaint, not a trust/leaving signal)
  - "Nobody came for my second booking, frustrating" -> N (single operational complaint, not repeated/severe)
  - "Please increase your service areas" -> N (a suggestion/request, not dissatisfaction)

Reviews to classify:
{reviews_lines}

Return ONLY a valid JSON array, no markdown formatting, no explanation, in this exact structure:
[{{"review_id": "...", "theme_1": "...", "theme_2_optional": null, "is_churn_risk": "Y"}}, ...]

Every review_id in the input must appear exactly once in your output."""
    return prompt


def extract_json(raw_text):
    """Strip markdown code fences if the model wraps its output in them."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def classify_batch(client, batch):
    prompt = build_prompt(batch)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            raw = extract_json(response.text)
            parsed = json.loads(raw)
            return parsed
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                wait_time = BASE_RETRY_DELAY_SECONDS * attempt  # 20s, 40s, 60s...
                print(f"  Waiting {wait_time}s before retrying (respecting rate limit)...")
                time.sleep(wait_time)
            else:
                print(f"  Giving up on this batch after {MAX_RETRIES} attempts.")
                return []


def validate_theme(theme):
    """
    GUARDRAIL: enforces the fixed taxonomy after generation, rather than
    trusting the model followed the prompt's instructions perfectly.

    LLMs frequently drift from a fixed list even when explicitly told not
    to -- in testing, ~6% of outputs used a category ('pricing_transparency')
    that had been removed from the taxonomy, because it remained
    semantically sensible to the model. Known aliases get remapped;
    anything else gets flagged for human review rather than silently
    accepted or dropped.
    """
    if theme is None:
        return theme

    normalized = str(theme).strip()
    if normalized == "" or normalized.lower() == "none":
        return None

    if normalized in TAXONOMY:
        return normalized

    if normalized in THEME_ALIASES:
        return THEME_ALIASES[normalized]

    return f"INVALID_THEME_NEEDS_REVIEW:{normalized}"


def load_reviews(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Run: export GEMINI_API_KEY='your-key-here'"
        )

    client = genai.Client(api_key=api_key)

    reviews = load_reviews(INPUT_FILE)
    print(f"Loaded {len(reviews)} reviews from {INPUT_FILE}")

    all_results = []
    sent_ids = set()

    for i in range(0, len(reviews), BATCH_SIZE):
        batch = reviews[i:i + BATCH_SIZE]
        batch_ids = {r["review_id"] for r in batch}
        sent_ids |= batch_ids

        print(f"Classifying batch {i // BATCH_SIZE + 1} ({len(batch)} reviews)...")
        results = classify_batch(client, batch)
        all_results.extend(results)

        time.sleep(SECONDS_BETWEEN_BATCHES)  # respect the 5 RPM account limit

    # Guardrail check: did every review_id we sent come back?
    returned_ids = {r["review_id"] for r in all_results}
    missing_ids = sent_ids - returned_ids

    if missing_ids:
        print(f"\nWARNING: {len(missing_ids)} review_ids were sent but never returned by the model.")
        print("Flagging these as MISSING_FROM_LLM_RESPONSE rather than silently dropping them.")
        for missing_id in missing_ids:
            all_results.append({
                "review_id": missing_id,
                "theme_1": "MISSING_FROM_LLM_RESPONSE",
                "theme_2_optional": None,
                "is_churn_risk": "MISSING_FROM_LLM_RESPONSE",
            })

    invalid_theme_count = 0
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["review_id", "theme_1", "theme_2_optional", "is_churn_risk"])
        writer.writeheader()
        for r in all_results:
            validated_theme_1 = validate_theme(r.get("theme_1"))
            validated_theme_2 = validate_theme(r.get("theme_2_optional"))

            if validated_theme_1 and validated_theme_1.startswith("INVALID_THEME_NEEDS_REVIEW"):
                invalid_theme_count += 1
            if validated_theme_2 and str(validated_theme_2).startswith("INVALID_THEME_NEEDS_REVIEW"):
                invalid_theme_count += 1

            writer.writerow({
                "review_id": r.get("review_id", ""),
                "theme_1": validated_theme_1,
                "theme_2_optional": validated_theme_2,
                "is_churn_risk": r.get("is_churn_risk", ""),
            })

    print(f"\nDone. Classified {len(all_results)} reviews -> {OUTPUT_FILE}")
    if missing_ids:
        print(f"({len(missing_ids)} flagged as missing -- review these manually)")
    if invalid_theme_count:
        print(f"GUARDRAIL: {invalid_theme_count} theme values were outside the fixed taxonomy and flagged as INVALID_THEME_NEEDS_REVIEW (aliases were auto-remapped, e.g. pricing_transparency -> over_priced).")


if __name__ == "__main__":
    main()
