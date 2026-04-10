"""
eval_offline.py — Offline evaluation of the OfficeQA agent.

Tests the agent against the local officeqa_full.csv dataset and reports
exact-match accuracy before submitting to the leaderboard.

Usage:
    # Test on 5 random questions (quick check):
    python scripts/eval_offline.py --sample 5

    # Test on all questions:
    python scripts/eval_offline.py

    # Test on specific UIDs:
    python scripts/eval_offline.py --uids UID0001,UID0010

Requirements:
    - Treasury data in /home/agent/treasury_data/ or ./treasury_data/
    - LLM provider configured via environment variables (see sample.env)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_offline")

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_eval_data(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def normalize_answer(answer: str) -> str:
    """Normalize for exact-match comparison."""
    a = answer.strip().lower()
    a = re.sub(r"[\$,]", "", a)
    a = re.sub(r"\s+", " ", a)
    a = a.rstrip("%").strip()
    # Remove trailing .0 for integers
    if re.match(r"^-?\d+\.0+$", a):
        a = a.split(".")[0]
    return a


def answers_match(predicted: str, expected: str) -> bool:
    """Check if predicted matches expected (flexible exact match)."""
    p = normalize_answer(predicted)
    e = normalize_answer(expected)
    if p == e:
        return True
    # Try numeric comparison
    try:
        pv = float(p.replace(",", ""))
        ev = float(e.replace(",", ""))
        if abs(pv - ev) < 0.01 * max(abs(ev), 1):
            return True
    except ValueError:
        pass
    return False


def run_eval(args):
    from agent import process_task
    from retrieval import _load_source_context

    csv_path = args.csv
    if not os.path.exists(csv_path):
        # Try relative to project root
        alt = Path(__file__).resolve().parent.parent.parent / "officeqa_full.csv"
        if alt.exists():
            csv_path = str(alt)
        else:
            logger.error(f"Cannot find eval CSV: {csv_path}")
            sys.exit(1)

    data = load_eval_data(csv_path)
    logger.info(f"Loaded {len(data)} questions from {csv_path}")

    if args.uids:
        uids = set(args.uids.split(","))
        data = [d for d in data if d["uid"] in uids]
    elif args.sample:
        data = random.sample(data, min(args.sample, len(data)))
    if args.difficulty:
        data = [d for d in data if d.get("difficulty") == args.difficulty]

    logger.info(f"Evaluating {len(data)} questions")

    results = []
    correct = 0
    total = 0

    for i, row in enumerate(data):
        uid = row["uid"]
        question = row["question"]
        expected = row["answer"]

        logger.info(f"\n[{i+1}/{len(data)}] {uid}: {question[:80]}...")
        t0 = time.time()

        try:
            response = process_task(question, task_type="officeqa")
            # Extract FINAL_ANSWER
            match = re.search(r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", response, re.DOTALL | re.IGNORECASE)
            predicted = match.group(1).strip() if match else response.strip().split("\n")[-1]
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            predicted = "ERROR"

        elapsed = time.time() - t0
        is_correct = answers_match(predicted, expected)
        if is_correct:
            correct += 1
        total += 1

        status = "CORRECT" if is_correct else "WRONG"
        logger.info(f"  Expected: {expected}")
        logger.info(f"  Got:      {predicted}")
        logger.info(f"  {status} ({elapsed:.1f}s)")

        results.append({
            "uid": uid,
            "question": question[:100],
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "elapsed_s": round(elapsed, 1),
            "difficulty": row.get("difficulty", ""),
        })

        if args.delay:
            time.sleep(args.delay)

    accuracy = correct / total if total > 0 else 0
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {correct}/{total} correct ({accuracy:.1%})")

    # Per-difficulty breakdown
    by_diff = {}
    for r in results:
        d = r["difficulty"] or "unknown"
        by_diff.setdefault(d, {"correct": 0, "total": 0})
        by_diff[d]["total"] += 1
        if r["correct"]:
            by_diff[d]["correct"] += 1
    for d, stats in sorted(by_diff.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        logger.info(f"  {d}: {stats['correct']}/{stats['total']} ({acc:.1%})")

    # Save results
    out_path = args.output or "eval_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total,
            "by_difficulty": by_diff,
            "results": results,
        }, f, indent=2)
    logger.info(f"Results saved to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Offline OfficeQA evaluation")
    p.add_argument("--csv", default="officeqa_full.csv", help="Path to eval CSV")
    p.add_argument("--sample", type=int, default=None, help="Random sample N questions")
    p.add_argument("--uids", type=str, default=None, help="Comma-separated UIDs to test")
    p.add_argument("--difficulty", type=str, default=None, help="Filter by difficulty (easy/hard)")
    p.add_argument("--delay", type=float, default=1.0, help="Delay between questions (rate limit)")
    p.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
