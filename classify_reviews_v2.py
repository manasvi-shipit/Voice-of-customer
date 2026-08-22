"""
classify_reviews.py

Stage 1 of the Voice of Customer Copilot agent pipeline (v2 -- universal
taxonomy).

Classifies reviews for ANY company (set via COMPANY_NAME below) against a
FIXED, domain-agnostic taxonomy. Classifying against a fixed taxonomy
(rather than open-ended clustering) means every prediction can be directly
compared to hand-labeled ground truth for precision/recall -- that
comparison is what turns this into a measurable eval, not just "the output
looks fine."

v1 (Urban-Company-only, 13 home-services-specific categories) is preserved
as classify_reviews.py. This v2 file generalizes the taxonomy to 12
domain-agnostic categories so the same pipeline can run against a
different company's reviews. See ground_truth_final_v2.csv for the
remapped ground truth this version evals against.

IMPORTANT: ground_truth_final_v2.csv only validates this taxonomy for
home-services-flavored (Urban Company) reviews. Running this against a
genuinely different company's reviews (food delivery, SaaS, e-commerce,
etc.) is UNVALIDATED until a small hand-labeled sample from that domain
confirms the taxonomy actually holds up there too.

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

INPUT_FILE = "custom_reviews.csv"  # dedicated file for any-company mode --
                                     # NEVER urban_company_reviews.csv, so a
                                     # different company's upload can never
                                     # overwrite or get mixed with the
                                     # validated Urban Company dataset
OUTPUT_FILE = "classified_reviews_v2.csv"  # separate from v1's output so the
                                             # validated Urban-Company-only run
                                             # (classified_reviews.csv) is never
                                             # overwritten

# The only thing that should need to change to classify a different
# company's reviews. Reads from the VOC_COMPANY_NAME environment variable
# so the UI (or a quick shell export) can set this per-run without editing
# this file. Falls back to "Urban Company" if unset.
COMPANY_NAME = os.environ.get("VOC_COMPANY_NAME", "Urban Company")

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

# Known aliases for categories that were merged/renamed over the project's
# history. If the model outputs one of these (because it's semantically
# sensible to it, even though the prompt no longer lists it as valid), we
# remap rather than silently accepting an out-of-taxonomy value.
THEME_ALIASES = {
    "pricing_transparency": "pricing_value",       # merged during Stage 1 eval iteration (Urban Company build)
    "over_priced": "pricing_value",                 # v1 (Urban-Company-specific) name -> v2 (universal) name
    "punctuality_noshow": "timeliness_reliability",  # v1 -> v2
    "booking_app_ux": "app_booking_ux",              # v1 -> v2
    "competitor's_praise": "competitor_comparison",  # v1 -> v2
    "new_service_ask": "feature_or_service_request", # v1 -> v2 (merged)
    "loyalty program ask": "feature_or_service_request",  # v1 -> v2 (merged)
    "useless review": "no_signal",                   # v1 -> v2
}

# UNIVERSAL TAXONOMY (v2). Generalized from the original Urban-Company-only
# taxonomy so this pipeline can classify reviews for any consumer product
# or service company, not just home services. Still a FIXED, closed list --
# genericizing the categories does not give up the fixed-taxonomy design
# choice (Section 14 of the project handoff): every prediction still maps
# to exactly one of these, so eval against hand-labeled ground truth still
# works. What changed is scope, not the fixed-vs-open-clustering decision.
#
# NOTE ON EVAL COVERAGE: ground_truth_final_v2.csv (remapped 1:1 from the
# original 92-row Urban Company ground truth) validates this taxonomy for
# home-services-flavored reviews. It does NOT by itself prove this taxonomy
# performs well on a genuinely different domain (e.g. food delivery, SaaS,
# e-commerce) -- that would need at least a small hand-labeled sample from
# that domain before trusting the numbers. Treat any other-company run as
# unvalidated until that eval work is done.
TAXONOMY = {
    "pricing_value": "Any pricing or value complaint -- cost too high, hidden fees, quoted-vs-charged mismatches, or poor value for money",
    "service_quality": "Quality of the core product or service delivered -- competence, workmanship, defects, redo/replacement requests",
    "timeliness_reliability": "Provider/order arriving late, not showing up/arriving, missed deadlines, or no communication about delays",
    "refund_cancellation": "Refund delays/denials or cancellation friction",
    "app_booking_ux": "Issues with the app or ordering/booking flow itself -- availability, scheduling, checkout, navigation",
    "safety_trust": "Safety concerns, inappropriate behavior, fraud, or trust/verification issues",
    "customer_support": "Poor or unresponsive customer support experience",
    "generic_praise": "Positive review with NO specific actionable detail (e.g. 'excellent service', 'good product')",
    "competitor_comparison": "Review explicitly compares this company unfavorably to a named competitor",
    "feature_or_service_request": "Review requests a new feature, service, offering, or loyalty/discount program that doesn't exist yet",
    "no_signal": "Review contains no real feedback signal at all -- not even generic praise (e.g. a stray, unrelated request)",
    "other_not_listed": "Doesn't fit any category above -- use sparingly, and only when genuinely nothing else fits",
}


def build_prompt(batch):
    taxonomy_lines = "\n".join(f'- "{name}": {desc}' for name, desc in TAXONOMY.items())
    reviews_lines = "\n".join(
        f'{{"review_id": "{r["review_id"]}", "rating": {r["rating"]}, "text": "{r["review_text"].replace(chr(34), chr(39))}"}}'
        for r in batch
    )

    prompt = f"""You are classifying customer reviews for {COMPANY_NAME} into a FIXED taxonomy. Do not invent new categories -- pick the closest fit from the list below, even if imperfect.

TAXONOMY (theme name: description):
{taxonomy_lines}

For each review below, assign:
- theme_1: the single best-fitting theme name from the taxonomy above (exact string match required)
- theme_2_optional: a second theme ONLY if the review clearly covers two distinct themes, otherwise null
- is_churn_risk: "Y" if the review signals real dissatisfaction that puts continued use of the app at risk -- this includes explicit statements of leaving/switching, but also strong statements like "never recommend," repeated severe operational failures described with frustration, or loss of trust in the platform. Simple negative adjectives ("bad," "worst," "pathetic") on their own, with no further context, are NOT sufficient -- but do not require an exact phrase match to the examples below; generalize the underlying signal.

  Examples of Y (real signal that the customer may stop using the platform):
  - "Not better than [a named competitor], switching to them from now on" -> Y (named competitor comparison)
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
