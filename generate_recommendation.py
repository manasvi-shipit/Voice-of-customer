"""
generate_recommendation.py

Stage 3 of the Voice of Customer Copilot agent pipeline.

Combines theme frequency, churn-risk density, and citation confidence
into a single prioritized roadmap recommendation. This is deliberately
NOT another LLM call -- the ranking logic is fully deterministic and
auditable, computed directly from Stage 1 and Stage 2 outputs. A PM
reading this should be able to see exactly why each theme ranked where
it did, not have to trust an opaque model judgment for prioritization.

PRIORITY SCORE = (review_count / total_reviews) * 0.5
                 + churn_rate * 0.5

Themes marked LOW_CONFIDENCE (Stage 2, <3 supporting reviews) are
excluded entirely and shown separately, since forcing a priority score
onto thin evidence would be misleading.

GUARDRAIL (this fix): themes with fewer than MIN_REVIEWS_FOR_RANKING
reviews are statistically unstable -- a tiny sample (e.g. 3 reviews,
100% churn) can outrank a much larger, more reliable theme (e.g. 17
reviews, 88% churn) purely because small-sample rates are noisy. These
themes are excluded from the main RANKED list and shown separately in
a WATCH_LIST section instead, so a small sample never outranks a
statistically robust one.

INPUT:  classified_reviews.csv, theme_report.csv
OUTPUT: prioritized_recommendation.csv, printed summary
"""

import csv

CLASSIFICATIONS_FILE = "classified_reviews.csv"
THEME_REPORT_FILE = "theme_report.csv"
OUTPUT_FILE = "prioritized_recommendation.csv"

FREQUENCY_WEIGHT = 0.5
CHURN_WEIGHT = 0.5

# GUARDRAIL: themes with fewer than this many reviews are excluded from
# the main ranked list (statistically unstable sample size) and shown
# in a separate watch list instead. Themes below MIN_REVIEWS_FOR_CITATION
# (handled upstream in Stage 2) are already excluded entirely as
# LOW_CONFIDENCE before they ever reach this script.
MIN_REVIEWS_FOR_RANKING = 10


def load_classifications():
    with open(CLASSIFICATIONS_FILE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_theme_report():
    with open(THEME_REPORT_FILE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_churn_rate_per_theme(classifications):
    theme_totals = {}
    theme_churn = {}

    for row in classifications:
        theme = (row.get("theme_1") or "").strip()
        if not theme:
            continue
        theme_totals[theme] = theme_totals.get(theme, 0) + 1
        if (row.get("is_churn_risk") or "").strip().upper() == "Y":
            theme_churn[theme] = theme_churn.get(theme, 0) + 1

    churn_rates = {}
    for theme, total in theme_totals.items():
        churn_rates[theme] = theme_churn.get(theme, 0) / total

    return churn_rates, theme_totals


def main():
    classifications = load_classifications()
    theme_reports = load_theme_report()
    total_reviews = len(classifications)

    churn_rates, theme_totals = compute_churn_rate_per_theme(classifications)

    scored = []
    low_confidence = []

    for report in theme_reports:
        theme = report["theme"]
        review_count = int(report["review_count"])
        confidence = report["confidence"]

        if confidence != "OK":
            low_confidence.append(report)
            continue

        churn_rate = churn_rates.get(theme, 0.0)
        frequency_share = review_count / total_reviews

        priority_score = (frequency_share * FREQUENCY_WEIGHT) + (churn_rate * CHURN_WEIGHT)

        scored.append({
            "theme": theme,
            "review_count": review_count,
            "frequency_share": round(frequency_share, 4),
            "churn_rate": round(churn_rate, 4),
            "priority_score": round(priority_score, 4),
            "summary": report["summary"],
        })

    ranked = [r for r in scored if r["review_count"] >= MIN_REVIEWS_FOR_RANKING]
    watch_list = [r for r in scored if r["review_count"] < MIN_REVIEWS_FOR_RANKING]

    ranked.sort(key=lambda x: -x["priority_score"])
    watch_list.sort(key=lambda x: -x["priority_score"])

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "section", "theme", "review_count", "frequency_share", "churn_rate", "priority_score", "summary"],
        )
        writer.writeheader()
        for i, r in enumerate(ranked, start=1):
            writer.writerow({"rank": i, "section": "RANKED", **r})
        for i, r in enumerate(watch_list, start=1):
            writer.writerow({"rank": i, "section": "WATCH_LIST", **r})

    print("=== PRIORITIZED ROADMAP RECOMMENDATION ===")
    print(f"(scored on {total_reviews} total classified reviews)\n")
    print(f"Main ranking includes themes with >= {MIN_REVIEWS_FOR_RANKING} reviews only.\n")

    for i, r in enumerate(ranked, start=1):
        print(f"#{i}. {r['theme']}  (score: {r['priority_score']})")
        print(f"    {r['review_count']} reviews ({r['frequency_share']:.1%} of all feedback) | churn rate: {r['churn_rate']:.1%}")
        print(f"    {r['summary'][:150]}...")
        print()

    if watch_list:
        print(f"=== WATCH LIST (fewer than {MIN_REVIEWS_FOR_RANKING} reviews -- shown, not ranked against the main list) ===")
        for r in watch_list:
            print(f"  - {r['theme']}: {r['review_count']} review(s), churn rate {r['churn_rate']:.1%} (score: {r['priority_score']}, informational only)")
        print()

    if low_confidence:
        print("=== EXCLUDED ENTIRELY (low confidence -- insufficient supporting reviews for citation) ===")
        for r in low_confidence:
            print(f"  - {r['theme']} ({r['review_count']} review(s)) -- {r['confidence']}")

    print(f"\nSaved full output to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
