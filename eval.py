"""
eval.py
-------
Retrieval quality evaluation for the Telugu RAG system.

Runs a curated set of Telugu questions against ChromaDB and prints
a detailed report showing:
  - Which sources were retrieved
  - Whether the expected article appeared in the top-3
  - A pass/fail grade per question
  - An overall accuracy score

Run:
    python eval.py

    # Run only the first N questions:
    python eval.py --limit 10

    # Save report to a file:
    python eval.py --out report.txt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from retriever import TeluguRetriever

# ---------------------------------------------------------------------------
# Test Suite
# Each entry: (question, expected_article_keyword_in_source_filename)
# The keyword is matched case-insensitively against the source filename.
# ---------------------------------------------------------------------------

TEST_QUESTIONS: list[tuple[str, str]] = [
    # ── రుద్రమ దేవి ──────────────────────────────────────────
    ("రుద్రమ దేవి ఎవరు?",                          "రుద్రమ"),
    ("రుద్రమ దేవి ఏ వంశానికి చెందినది?",           "రుద్రమ"),
    ("రుద్రమ దేవి పాలనా కాలం ఎంత?",               "రుద్రమ"),
    ("రుద్రమ దేవి అసలు పేరు ఏమిటి?",              "రుద్రమ"),
    ("రుద్రమ దేవి తండ్రి ఎవరు?",                   "రుద్రమ"),

    # ── కాకతీయ సామ్రాజ్యం ───────────────────────────────────
    ("కాకతీయ సామ్రాజ్య రాజధాని ఏది?",             "కాకతీయ"),
    ("కాకతీయులు ఎక్కడ పాలించారు?",                 "కాకతీయ"),
    ("కాకతీయ రాజవంశం ఎప్పుడు స్థాపించారు?",        "కాకతీయ"),
    ("కాకతీయుల చివరి రాజు ఎవరు?",                  "కాకతీయ"),
    ("రామప్ప దేవాలయం ఎవరు నిర్మించారు?",           "కాకతీయ"),

    # ── తెలంగాణ ─────────────────────────────────────────────
    ("తెలంగాణ ఎప్పుడు ఏర్పడింది?",                "తెలంగాణ"),
    ("తెలంగాణ రాజధాని ఏది?",                       "తెలంగాణ"),
    ("తెలంగాణ రాష్ట్రం ఏ సంవత్సరంలో ఏర్పడింది?", "తెలంగాణ"),
    ("తెలంగాణ జిల్లాలు ఎన్ని?",                   "తెలంగాణ"),
    ("తెలంగాణ ముఖ్యమంత్రి ఎవరు?",                 "తెలంగాణ"),

    # ── ఆంధ్రప్రదేశ్ ─────────────────────────────────────────
    ("ఆంధ్రప్రదేశ్ రాజధాని ఏది?",                 "ఆంధ్రప్రదేశ్"),
    ("ఆంధ్రప్రదేశ్ విభజన ఎప్పుడు జరిగింది?",      "ఆంధ్రప్రదేశ్"),
    ("ఆంధ్రప్రదేశ్ ముఖ్యమంత్రి ఎవరు?",            "ఆంధ్రప్రదేశ్"),
    ("ఆంధ్రప్రదేశ్ లో ఎన్ని జిల్లాలు ఉన్నాయి?",  "ఆంధ్రప్రదేశ్"),
    ("గోదావరి నది ఏ రాష్ట్రంలో ప్రవహిస్తుంది?",  "ఆంధ్రప్రదేశ్"),

    # ── పోతన ─────────────────────────────────────────────────
    ("పోతన ఎవరు?",                                 "పోతన"),
    ("పోతన రచించిన గ్రంథం ఏమిటి?",                "పోతన"),
    ("భాగవతం తెలుగులో ఎవరు రాశారు?",              "పోతన"),
    ("పోతన కాలం ఎప్పుడు?",                         "పోతన"),
    ("పోతన జన్మస్థలం ఏది?",                        "పోతన"),

    # ── నన్నయ ─────────────────────────────────────────────────
    ("నన్నయ రచనలు ఏమిటి?",                        "నన్నయ"),
    ("తెలుగు సాహిత్యంలో నన్నయ స్థానం ఏమిటి?",     "నన్నయ"),
    ("ఆదికవి ఎవరు?",                               "నన్నయ"),
    ("మహాభారతం తెలుగులో ఎవరు రాశారు?",            "నన్నయ"),
    ("నన్నయ ఏ కాలానికి చెందినవారు?",               "నన్నయ"),

    # ── తిక్కన ────────────────────────────────────────────────
    ("తిక్కన ఎవరు?",                               "తిక్కన"),
    ("తిక్కన రచించిన కావ్యాలు ఏమిటి?",             "తిక్కన"),
    ("కవిబ్రహ్మ అని ఎవరికి పేరు వచ్చింది?",        "తిక్కన"),
    ("తిక్కన కాలం ఏమిటి?",                         "తిక్కన"),
    ("మహాభారత రచనను పూర్తి చేసిన కవి ఎవరు?",      "తిక్కన"),

    # ── విజయనగర సామ్రాజ్యం ──────────────────────────────────
    ("విజయనగర సామ్రాజ్యం ఎక్కడ ఉంది?",            "విజయనగర"),
    ("విజయనగర సామ్రాజ్యం స్థాపకులు ఎవరు?",        "విజయనగర"),
    ("హంపి ఏ సామ్రాజ్యానికి రాజధాని?",             "విజయనగర"),
    ("విజయనగర సామ్రాజ్యం ఎప్పుడు పతనమైంది?",      "విజయనగర"),
    ("కృష్ణదేవరాయలు ఏ సామ్రాజ్యం వారు?",          "విజయనగర"),

    # ── శ్రీ కృష్ణదేవరాయలు ───────────────────────────────────
    ("శ్రీ కృష్ణదేవరాయలు ఎవరు?",                  "కృష్ణదేవరాయ"),
    ("కృష్ణదేవరాయలు పాలించిన కాలం ఏది?",          "కృష్ణదేవరాయ"),
    ("అముక్తమాల్యద ఎవరు రాశారు?",                  "కృష్ణదేవరాయ"),
    ("అష్టదిగ్గజాలు ఎవరు?",                        "కృష్ణదేవరాయ"),
    ("తెలుగు సాహిత్య స్వర్ణయుగం ఏది?",            "కృష్ణదేవరాయ"),

    # ── అమరావతి ──────────────────────────────────────────────
    ("అమరావతి ఎక్కడ ఉంది?",                       "అమరావతి"),
    ("అమరావతి స్తూపం గురించి చెప్పండి?",           "అమరావతి"),
    ("అమరావతి రాజధాని ఎవరు నిర్మించారు?",         "అమరావతి"),

    # ── వరంగల్ ───────────────────────────────────────────────
    ("వరంగల్ ఎక్కడ ఉంది?",                        "వరంగల్"),
    ("వరంగల్ కోట గురించి చెప్పండి?",              "వరంగల్"),
    ("వరంగల్ జిల్లా ప్రత్యేకత ఏమిటి?",             "వరంగల్"),
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    question: str
    expected_keyword: str
    retrieved_sources: list[str]
    retrieved_titles: list[str]
    top_content_preview: str
    passed: bool


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def keyword_in_sources(keyword: str, sources: list[str]) -> bool:
    """
    Returns True if the expected keyword appears in any of the
    retrieved source filenames (case-insensitive Unicode match).
    """
    kw = keyword.lower()
    return any(kw in src.lower() for src in sources)


def run_evaluation(
    retriever: TeluguRetriever,
    questions: list[tuple[str, str]],
) -> list[EvalResult]:
    """
    Runs every question through the retriever and builds EvalResult objects.

    Args:
        retriever: Initialised TeluguRetriever instance.
        questions: List of (question, expected_keyword) tuples.

    Returns:
        List of EvalResult, one per question.
    """
    results: list[EvalResult] = []

    for question, expected_keyword in questions:
        chunks = retriever.search(question, k=3)

        sources = [c.source for c in chunks]
        titles  = [c.title  for c in chunks]
        preview = chunks[0].content[:120].replace("\n", " ") if chunks else "(no results)"

        passed = keyword_in_sources(expected_keyword, sources)

        results.append(EvalResult(
            question=question,
            expected_keyword=expected_keyword,
            retrieved_sources=sources,
            retrieved_titles=titles,
            top_content_preview=preview,
            passed=passed,
        ))

    return results


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

PASS_ICON = "✅"
FAIL_ICON = "❌"
SEPARATOR = "─" * 72


def render_report(results: list[EvalResult], stream=sys.stdout) -> None:
    """Prints a human-readable evaluation report to stream."""

    def out(line: str = "") -> None:
        print(line, file=stream)

    passed_count = sum(1 for r in results if r.passed)
    total = len(results)
    accuracy = (passed_count / total * 100) if total else 0

    out()
    out("=" * 72)
    out("  Telugu RAG — Retrieval Quality Evaluation Report")
    out("=" * 72)
    out(f"  Questions tested : {total}")
    out(f"  Passed           : {passed_count}  {PASS_ICON}")
    out(f"  Failed           : {total - passed_count}  {FAIL_ICON}")
    out(f"  Accuracy         : {accuracy:.1f}%")
    out("=" * 72)

    # ── Per-question breakdown ─────────────────────────────────────────────
    for i, r in enumerate(results, start=1):
        icon = PASS_ICON if r.passed else FAIL_ICON
        out()
        out(f"{icon} [{i:02d}] {r.question}")
        out(f"       Expected keyword : {r.expected_keyword}")
        out(f"       Retrieved sources: {', '.join(r.retrieved_sources)}")
        out(f"       Retrieved titles : {', '.join(r.retrieved_titles)}")
        out(f"       Top chunk preview: {r.top_content_preview!r}")
        out(SEPARATOR)

    # ── Failures summary ───────────────────────────────────────────────────
    failures = [r for r in results if not r.passed]
    if failures:
        out()
        out("── FAILED QUESTIONS (review these) ──────────────────────────────────")
        for r in failures:
            out(f"  {FAIL_ICON} {r.question}")
            out(f"       Expected : {r.expected_keyword}")
            out(f"       Got      : {', '.join(r.retrieved_sources)}")
        out()

    # ── Final verdict ──────────────────────────────────────────────────────
    out()
    if accuracy >= 90:
        verdict = "🟢 EXCELLENT — Retrieval quality is strong. Safe to add LLM."
    elif accuracy >= 75:
        verdict = "🟡 GOOD — Minor retrieval gaps. Consider adding more article text."
    elif accuracy >= 60:
        verdict = "🟠 FAIR — Some retrieval gaps. Review chunk size or add data."
    else:
        verdict = "🔴 POOR — Retrieval needs improvement before adding an LLM."

    out(f"  Verdict: {verdict}")
    out()
    out("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Telugu RAG retrieval quality"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N questions (default: all)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Save report to this file in addition to stdout"
    )
    args = parser.parse_args()

    # Slice the test suite if --limit is set
    questions = TEST_QUESTIONS[: args.limit] if args.limit else TEST_QUESTIONS

    print(f"\nInitialising retriever …")
    retriever = TeluguRetriever()

    print(f"Running {len(questions)} evaluation question(s) …\n")
    results = run_evaluation(retriever, questions)

    # Always print to stdout
    render_report(results, stream=sys.stdout)

    # Optionally save to file
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            render_report(results, stream=f)
        print(f"\nReport saved to: {args.out}")


if __name__ == "__main__":
    main()
