# -*- coding: utf-8 -*-
"""
HONEYPOT HUNTER — find the ~80 deliberately impossible profiles (forced tier 0).

Method: run every consistency check available, keep only checks whose violators
form a sharply separated "island" from the population (honeypots are ~80/100k,
so a true marker must flag tens of candidates, not thousands). Checks that
flag thousands are generator noise -> classified SOFT, not honeypot evidence.

VALIDATED HARD MARKERS (each an island, physically impossible):
  A expert_zero_months : 3-5 skills at "expert" proficiency with 0 months used
  B date_mismatch      : claimed job duration_months differs from start/end
                         dates by 4-189 months (population max otherwise ~2)
  C yoe_vs_span        : profile yoe exceeds entire career span by 8-12.6 yrs
  D summary_yoe        : yoe stated in summary text differs from yoe field by
                         1.7-12.6 yrs (population otherwise <= 0.7, mostly 0)
  E cert_2030          : "AWS Certified ML Specialty" dated 2030 (future;
                         all other cert years are 2018-2025)

REJECTED AS NOISE (violate realism but affect 1k-30k candidates -> generator
artifacts, cannot be tier-0 markers):
  - job starts 1-5y before real company founding (CRED/Sarvam AI/Krutrim...)
  - skill duration > tech release age (LangChain 96mo etc.)   ~1,076 cands
  - bachelor's starting after master's/PhD ended               6,121
  - career starting 8+ yrs after last education ended         30,531
  - signup_date after last_active_date (max gap 234d)          7,496
  - salary min > max                                          18,865
  - saved_by_recruiters > profile_views                        7,677
  - assessment scores for unlisted skills                         36
  (the 36 unlisted-assessment cands look like noise too: scores ~ same dist
   as listed, 1 overlap with hard set, mixed mundane titles)

Run:  PYTHONIOENCODING=utf-8 python eda/honeypots.py
"""
import json
import os
import re
from datetime import date

import numpy as np
import pandas as pd

TODAY = date(2026, 6, 10)
ROOT = "c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons"
PARQUET = f"{ROOT}/redrob-ranker/data/candidates_flat.parquet"
JSONL = (f"{ROOT}/hack2skill/The Data & AI Challenge/"
         "[PUB] India_runs_data_and_ai_challenge/"
         "India_runs_data_and_ai_challenge/candidates.jsonl")
CACHE = f"{ROOT}/redrob-ranker/eda/honeypot_jsonl_cache_v2.parquet"
OUT_LIST = f"{ROOT}/redrob-ranker/eda/honeypot_candidates.csv"

FOUNDING = {  # real founding years; used only for the (rejected) soft check
    "sarvam ai": 2023, "krutrim": 2023, "glance": 2019, "rephrase.ai": 2019,
    "cred": 2018, "observe.ai": 2017, "niramai": 2017, "saarthi.ai": 2017,
    "yellow.ai": 2016, "verloop.io": 2016, "meesho": 2015, "unacademy": 2015,
    "pharmeasy": 2015, "upgrad": 2015, "phonepe": 2015, "wysa": 2015,
    "locobuzz": 2015, "swiggy": 2014, "razorpay": 2014, "haptik": 2013,
    "mad street den": 2013, "nykaa": 2012, "byju's": 2011, "vedantu": 2011,
    "paytm": 2010, "ola": 2010, "freshworks": 2010, "zomato": 2008,
    "dream11": 2008, "policybazaar": 2008, "aganitha": 2008, "inmobi": 2007,
    "flipkart": 2007,
}


def parse_date(s):
    if not s:
        return None
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def months_between(a, b):
    return (b - a).days / 30.4375


def jsonl_pass():
    """Single pass over raw JSONL for nested-only fields."""
    rows = []
    with open(JSONL, encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            jobs = r.get("career_history") or []
            max_mismatch = 0.0
            n_founding_viol = 0
            for j in jobs:
                sd = parse_date(j.get("start_date"))
                ed = parse_date(j.get("end_date"))
                dur = j.get("duration_months")
                if sd and dur is not None:
                    eff = TODAY if (j.get("is_current") or ed is None) else ed
                    max_mismatch = max(max_mismatch,
                                       abs(dur - months_between(sd, min(eff, TODAY))))
                fy = FOUNDING.get((j.get("company") or "").strip().lower())
                if sd and fy and sd.year < fy:
                    n_founding_viol += 1
            cert_years = [c.get("year") for c in (r.get("certifications") or [])
                          if c.get("year")]
            rows.append(dict(
                candidate_id=r["candidate_id"],
                max_date_mismatch=round(max_mismatch, 1),
                n_founding_viol=n_founding_viol,
                cert_max_year=max(cert_years) if cert_years else 0,
            ))
            if (i + 1) % 25000 == 0:
                print(f"  jsonl pass: {i+1}", flush=True)
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(PARQUET)
    print(f"loaded parquet: {df.shape}")
    if os.path.exists(CACHE):
        jx = pd.read_parquet(CACHE)
    else:
        print("raw JSONL pass (one-time, ~2 min)...")
        jx = jsonl_pass()
        jx.to_parquet(CACHE)
    df = df.merge(jx, on="candidate_id", how="left")

    # yoe stated in summary text (99,992/100k parse cleanly)
    pat = re.compile(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)(?:'| of)?\s*(?:hands-on\s+)?experience")
    df["summary_yoe"] = [float(m.group(1)) if (m := pat.search(s)) else np.nan
                         for s in df.summary.fillna("")]

    # ---------------- HARD markers (islands, physically impossible) ------
    H = {
        "A_expert_zero_months": df.n_expert_zero_months > 0,
        "B_date_mismatch": df.max_date_mismatch > 3,
        "C_yoe_vs_span": df.yoe * 12 > df.career_span_months + 24,
        "D_summary_yoe_gap": (df.summary_yoe - df.yoe).abs() > 1.5,
        "E_cert_future": df.cert_max_year > 2026,
    }
    # ---------------- SOFT anomalies (generator noise, NOT honeypots) ----
    S = {
        "founding_violation": df.n_founding_viol > 0,
        "signup_after_active": df.signup_after_active,
        "salary_min_gt_max": df.sig_salary_min > df.sig_salary_max,
        "saved_gt_views": df.sig_saved_by_recruiters_30d > df.sig_profile_views_30d,
    }

    print("\n=== HARD markers ===")
    for k, m in H.items():
        print(f"  {k:24s} {int(m.sum()):5d}")
    print("=== SOFT anomalies (noise; do NOT use as honeypot evidence) ===")
    for k, m in S.items():
        print(f"  {k:24s} {int(m.sum()):5d}")

    # island evidence
    print("\nisland evidence:")
    print(f"  B: population max_date_mismatch p99.9 = "
          f"{df.max_date_mismatch.quantile(0.999):.1f}m; flagged range "
          f"{df.loc[H['B_date_mismatch'], 'max_date_mismatch'].min():.1f}-"
          f"{df.loc[H['B_date_mismatch'], 'max_date_mismatch'].max():.1f}m")
    g = (df.summary_yoe - df.yoe).abs()
    print(f"  D: gaps > 0.2y: {int((g > 0.2).sum())}; gaps in (0.2,1.5]: "
          f"{int(((g > 0.2) & (g <= 1.5)).sum())}; flagged min gap "
          f"{g[H['D_summary_yoe_gap']].min():.1f}y")
    exc = (df.yoe * 12 - df.career_span_months)[H["C_yoe_vs_span"]]
    print(f"  C: flagged yoe excess {exc.min():.0f}-{exc.max():.0f} months")
    print(f"  E: cert years observed: "
          f"{sorted(df.cert_max_year[df.cert_max_year > 0].unique())}")

    names = list(H)
    hm = np.column_stack([H[k].values for k in names])
    df["n_hp_flags"] = hm.sum(axis=1)
    print("\noverlap matrix:")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov = int((H[a] & H[b]).sum())
            if ov:
                print(f"  {a} & {b}: {ov}")
    print("\nflag-count histogram (all 100k):")
    print(df.n_hp_flags.value_counts().sort_index().to_string())

    hp = df[df.n_hp_flags > 0].copy()
    hp["families"] = [",".join(k for k, v in zip(names, row) if v)
                      for row in hm[df.n_hp_flags > 0]]
    print(f"\nFINAL HONEYPOT SET: {len(hp)} candidates")
    print(hp.families.value_counts().to_string())

    # which would actually threaten a top-100 (AI-bait titles)?
    bait = hp.current_title.str.contains(
        "AI|ML|NLP|Search|Applied|Machine|Recommendation|Data Scien",
        regex=True)
    print(f"\nAI-bait titles among honeypots: {int(bait.sum())}")
    print(hp.loc[bait, ["candidate_id", "families", "yoe", "summary_yoe",
                        "career_span_months", "current_title",
                        "current_company"]].to_string(index=False))

    cols = ["candidate_id", "n_hp_flags", "families", "yoe", "summary_yoe",
            "career_span_months", "max_date_mismatch", "n_expert_zero_months",
            "cert_max_year", "current_title", "current_company"]
    hp.sort_values(["n_hp_flags", "candidate_id"],
                   ascending=[False, True])[cols].to_csv(OUT_LIST, index=False)
    print(f"\nwrote {len(hp)} rows -> {OUT_LIST}")


if __name__ == "__main__":
    main()
