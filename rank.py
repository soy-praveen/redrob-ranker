#!/usr/bin/env python3
"""Redrob candidate ranker: candidates.jsonl -> submission.csv (top 100).

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Runs end-to-end on CPU with no network access; ~1 minute for 100k candidates.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from features import build_features
from reasoning import generate_reasoning
from scoring import HONEYPOT_MARKERS, load_params, rank_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default=str(Path(__file__).parent / "config" / "params.json"))
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/4] extracting features from {args.candidates} ...")
    df = build_features(args.candidates)
    print(f"      {len(df)} candidates, {time.time() - t0:.1f}s")

    print("[2/4] scoring ...")
    params = load_params(args.params)
    scored, ranked = rank_pool(df, params, top_n=args.top_n)
    n_excl = (scored["exclusion"] != "").sum()
    print(f"      excluded {n_excl} (honeypots/stuffers/clones/non-tech), {time.time() - t0:.1f}s")

    print("[3/4] generating reasoning ...")
    grades = params["para_grades"]
    tags = params["evidence_tags"]
    ranked["reasoning"] = [
        generate_reasoning(row, row["rank"], grades, tags)
        for _, row in ranked.iterrows()
    ]

    print("[4/4] assembly checks ...")
    assert len(ranked) == args.top_n, f"expected {args.top_n} rows, got {len(ranked)}"
    assert ranked["candidate_id"].is_unique, "duplicate candidate ids"
    assert (ranked[HONEYPOT_MARKERS].sum().sum() == 0), "honeypot marker present in output!"
    assert not (ranked["family"] == "STUFFER").any(), "stuffer in output!"
    assert ranked["score"].is_monotonic_decreasing, "scores must be non-increasing"
    # equal-score tie-break must be candidate_id ascending (validator rule)
    for i in range(len(ranked) - 1):
        if ranked["score"].iat[i] == ranked["score"].iat[i + 1]:
            assert ranked["candidate_id"].iat[i] < ranked["candidate_id"].iat[i + 1]
    india_share = (ranked["country"].str.lower() == "india").mean()
    print(f"      india_share={india_share:.0%}, "
          f"grade5={(ranked['best_grade'] == 5).sum()}, grade4={(ranked['best_grade'] == 4).sum()}, "
          f"grade3={(ranked['best_grade'] == 3).sum()}")

    out = ranked[["candidate_id", "rank", "score", "reasoning"]]
    out.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")
    print(f"done: {args.out} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
