"""
generate_recommendation_v2.py

Stage 3 of the Voice of Customer Copilot agent pipeline (v2 -- paired with
the universal taxonomy / any-company mode). No logic changes from the
validated v1 script -- theme names are read dynamically, nothing
taxonomy-specific is hardcoded here. Only filenames changed, to keep this
run separate from the validated Urban Company v1 output.

Combines theme frequency, churn-risk density, and citation confidence
into a single prioritized roadmap recommendation. This is deliberately
NOT another LLM call -- the ranking logic is fully deterministic and
auditable, computed directly from Stage 1 and Stage 2 outputs.

INPUT:  classified_reviews_v2.csv, theme_report_v2.csv
OUTPUT: prioritized_recommendation_v2.csv, printed summary
"""

import csv

CLASSIFICATIONS_FILE = "classified_reviews_v2.csv"
THEME_REPORT_FILE = "theme_report_v2.csv"
OUTPUT_FILE = "prioritized_recommendation_v2.csv"

FREQUENCY_WEIGHT = 0.5
CHURN_WEIGHT = 0.5

# GUARDRAIL: themes with too few reviews are excluded from the main ranked
# list (statistically unstable sample size) and shown in a watch list
# instead. Unlike v1 (Urban Company), which uses a fixed floor of 10
# because that was validated against a specific 376-review dataset, this
# threshold SCALES with the actual dataset size: any-company mode can see
# anywhere from a few dozen reviews (a quick test) to thousands (a real
# scrape), and a fixed floor of 10 is far too strict for a 40-review run
# (12 categories x 40 reviews averages ~3 reviews/theme -- almost nothing
# would ever qualify) while potentially too loose for a much larger one.
# Rule: 10% of the dataset, with a floor of MIN_REVIEWS_FOR_CITATION (3) --
# never rank on a sample so thin that Stage 2 wouldn't even cite it.
MIN_REVIEWS_FOR_CITATION = 3  # mirrors Stage 2's threshold, used as the floor here
RANKING_THRESHOLD_FRACTION = 0.10


def compute_ranking_threshold(total_reviews):
    return max(MIN_REVIEWS_FOR_CITATION, round(total_reviews * RANKING_THRESHOLD_FRACTION))


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
    min_reviews_for_ranking = compute_ranking_threshold(total_reviews)

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

    # GUARDRAIL: split into statistically robust RANKED themes vs a
    # small-sample WATCH_LIST. Sort each independently -- watch list
    # themes are informative but never compete against ranked ones.
    ranked = [r for r in scored if r["review_count"] >= min_reviews_for_ranking]
    watch_list = [r for r in scored if r["review_count"] < min_reviews_for_ranking]

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
    print(f"Ranking threshold for this run: >= {min_reviews_for_ranking} reviews "
          f"(10% of {total_reviews}, floored at {MIN_REVIEWS_FOR_CITATION}).\n")

    for i, r in enumerate(ranked, start=1):
        print(f"#{i}. {r['theme']}  (score: {r['priority_score']})")
        print(f"    {r['review_count']} reviews ({r['frequency_share']:.1%} of all feedback) | churn rate: {r['churn_rate']:.1%}")
        print(f"    {r['summary'][:150]}...")
        print()

    if watch_list:
        print(f"=== WATCH LIST (fewer than {min_reviews_for_ranking} reviews -- shown, not ranked against the main list) ===")
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
