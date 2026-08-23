"""Analysis for the Clean Forcing 2AFC human-preference study.

Usage:
    python analysis.py --responses responses.csv [--manifest manifest.csv]
                       [--per-prompt-csv per_prompt.csv]
    python analysis.py --selftest

Input responses CSV columns: rater_id, pair_id, question, answer
  question in {q_overall, q_stability}, answer in {A, B}.

Pipeline:
1. Join responses to manifest.csv (the unblinding key; local only).
2. Exclude every rater who fails ANY catch trial. A catch is failed when the
   rater's q_overall answer for a catch pair differs from
   expected_catch_answer (the side showing ours; the other side is a fully
   collapsed adapted-base video, so the primary question has an objectively
   correct answer).
3. For non-catch rows, per pairing x question: preference % for ours,
   Wilson 95% CI, two-sided exact binomial p vs 0.5.
4. Per-prompt breakdown: ours wins / n per (pairing, prompt_idx, question).
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

QUESTIONS = ["q_overall", "q_stability"]
QUESTION_TEXT = {
    "q_overall": "Which video is better overall, considering visual quality "
                 "and alignment with the prompt?",
    "q_stability": "Which video stays cleaner and more stable over time?",
}
CATCH_QUESTION = "q_overall"
# Catch items voided 2026-08-17 after item analysis of the first Prolific run:
# failed by ~50% of raters who passed everything else (chance-level -> the item
# does not discriminate attention; collapsed side likely reads as "stylized").
# Working catches (pair_012/015/018/056) failed at 0-10%. Voided items still
# play in the survey but are not graded.
VOID_CATCHES = {"pair_009", "pair_076"}


def wilson_ci(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def binomial_two_sided_p(wins: int, n: int) -> float:
    """Exact two-sided binomial test vs p=0.5 (minlike method, matches
    scipy.stats.binomtest for p=0.5)."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, k) * 0.5 ** n for k in range(n + 1)]
    threshold = pmf[wins] * (1 + 1e-12)
    return min(1.0, sum(p for p in pmf if p <= threshold))


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def validate_responses(responses: list[dict]) -> None:
    required = {"rater_id", "pair_id", "question", "answer"}
    if not responses:
        raise ValueError("responses CSV is empty")
    missing = required - set(responses[0].keys())
    if missing:
        raise ValueError(f"responses CSV missing columns: {sorted(missing)}")
    for i, row in enumerate(responses):
        if row["answer"] not in ("A", "B"):
            raise ValueError(f"row {i + 2}: answer must be A or B, "
                             f"got {row['answer']!r}")
        if row["question"] not in QUESTIONS:
            raise ValueError(f"row {i + 2}: unknown question "
                             f"{row['question']!r} (expected {QUESTIONS})")


def failing_raters(responses: list[dict], manifest: dict[str, dict]) -> set[str]:
    failed = set()
    for row in responses:
        pair = manifest[row["pair_id"]]
        if (pair["is_catch"] == "True" and row["pair_id"] not in VOID_CATCHES
                and row["question"] == CATCH_QUESTION
                and row["answer"] != pair["expected_catch_answer"]):
            failed.add(row["rater_id"])
    return failed


def ours_side(pair: dict) -> str:
    if pair["left_system"] == "ours":
        return "A"
    if pair["right_system"] == "ours":
        return "B"
    raise ValueError(f"pair {pair['pair_id']} has no 'ours' side")


def analyze(responses: list[dict], manifest_rows: list[dict],
            per_prompt_csv: Path | None = None) -> dict:
    manifest = {r["pair_id"]: r for r in manifest_rows}
    validate_responses(responses)
    unknown = {r["pair_id"] for r in responses} - set(manifest)
    if unknown:
        raise ValueError(f"responses reference unknown pair_ids: {sorted(unknown)}")

    excluded = failing_raters(responses, manifest)
    kept = [r for r in responses if r["rater_id"] not in excluded]

    # (pairing, question) -> [ours_wins, n];  (pairing, prompt, question) -> same
    agg = defaultdict(lambda: [0, 0])
    per_prompt = defaultdict(lambda: [0, 0])
    for row in kept:
        pair = manifest[row["pair_id"]]
        if pair["is_catch"] == "True":
            continue
        win = int(row["answer"] == ours_side(pair))
        for key in ((pair["pairing"], row["question"]),):
            agg[key][0] += win
            agg[key][1] += 1
        pkey = (pair["pairing"], int(pair["prompt_idx"]), row["question"])
        per_prompt[pkey][0] += win
        per_prompt[pkey][1] += 1

    results = {}
    for (pairing, question), (wins, n) in sorted(agg.items()):
        lo, hi = wilson_ci(wins, n)
        results[(pairing, question)] = {
            "wins": wins, "n": n,
            "pct": 100.0 * wins / n if n else float("nan"),
            "ci_lo": 100.0 * lo, "ci_hi": 100.0 * hi,
            "p": binomial_two_sided_p(wins, n),
        }

    n_raters = len({r["rater_id"] for r in responses})
    print(f"raters: {n_raters} total, {len(excluded)} excluded for failing "
          f"catch trials ({sorted(excluded)})")
    print(f"pair-ratings kept (non-catch): "
          f"{sum(v[1] for v in agg.values())}\n")
    header = (f"{'pairing':<14} {'question':<12} {'ours%':>7} "
              f"{'wilson95':>17} {'n':>5} {'p(two-sided)':>13}")
    print(header)
    print("-" * len(header))
    for (pairing, question), r in sorted(results.items()):
        print(f"{pairing:<14} {question:<12} {r['pct']:>6.1f}% "
              f"[{r['ci_lo']:>5.1f}%,{r['ci_hi']:>6.1f}%] {r['n']:>5} "
              f"{r['p']:>13.4g}")

    print("\nper-prompt breakdown (ours wins / ratings):")
    print(f"{'pairing':<14} {'prompt':>6} " +
          " ".join(f"{q:>12}" for q in QUESTIONS))
    prompts = sorted({(p, i) for (p, i, _) in per_prompt})
    for pairing, idx in prompts:
        cells = []
        for q in QUESTIONS:
            wins, n = per_prompt.get((pairing, idx, q), (0, 0))
            cells.append(f"{wins}/{n}" if n else "-")
        print(f"{pairing:<14} {idx:>6} " +
              " ".join(f"{c:>12}" for c in cells))

    if per_prompt_csv:
        with open(per_prompt_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pairing", "prompt_idx", "question", "ours_wins", "n"])
            for (pairing, idx, q), (wins, n) in sorted(per_prompt.items()):
                w.writerow([pairing, idx, q, wins, n])
        print(f"\nper-prompt CSV written to {per_prompt_csv}")

    return {"results": results, "excluded": excluded,
            "per_prompt": dict(per_prompt)}


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _synthetic_manifest() -> list[dict]:
    rows = []
    for i, (pairing, prompt_idx, is_catch) in enumerate(
            [("ours_vs_sf", 0, False), ("ours_vs_sf", 5, False),
             ("ours_vs_base", 0, False), ("ours_vs_base", 5, False),
             ("ours_vs_sf", 18, True)], start=1):
        left = "ours" if i % 2 else "adapted_base"
        right = "adapted_base" if i % 2 else "ours"
        rows.append({
            "pair_id": f"pair_{i:03d}", "pairing": pairing,
            "prompt_idx": str(prompt_idx), "prompt_text": "synthetic",
            "left_system": left, "right_system": right,
            "is_catch": str(is_catch),
            "expected_catch_answer": ("A" if left == "ours" else "B") if is_catch else "",
        })
    return rows


def selftest() -> None:
    # Wilson CI against reference values (statsmodels, wilson, 95%)
    lo, hi = wilson_ci(8, 10)
    assert abs(lo - 0.4901625) < 1e-6 and abs(hi - 0.9433178) < 1e-6, (lo, hi)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775328) < 1e-6, (lo, hi)

    # Exact two-sided binomial vs 0.5: 9/10 -> 2*11/1024
    assert abs(binomial_two_sided_p(9, 10) - 0.021484375) < 1e-12
    assert abs(binomial_two_sided_p(5, 10) - 1.0) < 1e-12
    assert abs(binomial_two_sided_p(10, 10) - 2 / 1024) < 1e-15
    try:
        from scipy.stats import binomtest
        for wins, n in [(9, 10), (13, 20), (60, 100), (7, 13)]:
            expect = binomtest(wins, n, 0.5).pvalue
            got = binomial_two_sided_p(wins, n)
            assert abs(got - expect) < 1e-9, (wins, n, got, expect)
    except ImportError:
        print("  (scipy unavailable; skipped cross-check)")

    manifest = _synthetic_manifest()
    # ours is LEFT (A) on pair_001/003/005(catch), RIGHT (B) on pair_002/004.
    ours_answer = {"pair_001": "A", "pair_002": "B",
                   "pair_003": "A", "pair_004": "B", "pair_005": "A"}
    responses = []

    def vote(rater, pair, ours_pref: dict[str, bool]):
        for q in QUESTIONS:
            ans = ours_answer[pair] if ours_pref[q] else \
                ("B" if ours_answer[pair] == "A" else "A")
            responses.append({"rater_id": rater, "pair_id": pair,
                              "question": q, "answer": ans})

    # 3 good raters: prefer ours on everything, pass the catch.
    for rater in ["r1", "r2", "r3"]:
        for pair in ["pair_001", "pair_002", "pair_003", "pair_004", "pair_005"]:
            vote(rater, pair, {q: True for q in QUESTIONS})
    # r4 fails the catch (answers against ours on the catch pair) but prefers
    # ours everywhere else -> must be excluded entirely.
    for pair in ["pair_001", "pair_002", "pair_003", "pair_004"]:
        vote("r4", pair, {q: True for q in QUESTIONS})
    vote("r4", "pair_005", {q: False for q in QUESTIONS})
    # r5 passes the catch but prefers baseline on q_overall for ours_vs_sf.
    vote("r5", "pair_001", {"q_overall": False, "q_stability": True})
    vote("r5", "pair_002", {"q_overall": False, "q_stability": True})
    vote("r5", "pair_003", {q: True for q in QUESTIONS})
    vote("r5", "pair_004", {q: True for q in QUESTIONS})
    vote("r5", "pair_005", {q: True for q in QUESTIONS})

    out = analyze(responses, manifest)
    assert out["excluded"] == {"r4"}, out["excluded"]

    r = out["results"]
    # ours_vs_sf q_overall: r1-r3 all ours (6 wins), r5 0/2 -> 6/8
    assert (r[("ours_vs_sf", "q_overall")]["wins"],
            r[("ours_vs_sf", "q_overall")]["n"]) == (6, 8)
    # ours_vs_sf q_stability: all 4 kept raters prefer ours -> 8/8
    assert (r[("ours_vs_sf", "q_stability")]["wins"],
            r[("ours_vs_sf", "q_stability")]["n"]) == (8, 8)
    # ours_vs_base: 4 kept raters x 2 pairs, all ours -> 8/8 on each question
    for q in QUESTIONS:
        assert (r[("ours_vs_base", q)]["wins"], r[("ours_vs_base", q)]["n"]) == (8, 8)
    # catch rows contribute nothing to preference stats
    assert sum(v["n"] for v in r.values()) == len(QUESTIONS) * 2 * 8
    # per-prompt: ours_vs_sf prompt 0 q_overall = 3/4
    assert out["per_prompt"][("ours_vs_sf", 0, "q_overall")] == [3, 4]

    # answer validation catches bad input
    try:
        analyze([{"rater_id": "x", "pair_id": "pair_001",
                  "question": "q_overall", "answer": "C"}], manifest)
        raise AssertionError("expected ValueError for bad answer")
    except ValueError:
        pass

    print("\nSELFTEST PASSED")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses", type=Path,
                    help="CSV: rater_id,pair_id,question,answer")
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).parent / "manifest.csv")
    ap.add_argument("--per-prompt-csv", type=Path, default=None,
                    help="optional path to write the per-prompt breakdown")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0
    if not args.responses:
        ap.error("--responses is required (or use --selftest)")
    analyze(load_csv(args.responses), load_csv(args.manifest),
            per_prompt_csv=args.per_prompt_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
