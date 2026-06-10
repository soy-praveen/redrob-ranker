# -*- coding: utf-8 -*-
"""
T3 deep-dive: are pre-founding tenures at young companies PLANTED TRAPS or
GENERATOR NOISE? Discriminators:
  1. pool-wide start-year distribution at each young company (founding-aware?)
  2. violation rate conditional on having a tenure there: AI vs non-AI,
     grade>=4 vs not (traps would concentrate in attractive candidates)
  3. README-literal check: tenure DURATION > company age (+12mo slack)
  4. internal consistency (5 hard markers) of the worst worthy suspects
  5. E-family sanity: prevalence of 'AWS Certified ML Specialty' pool-wide
Run: PYTHONIOENCODING=utf-8 python eda/round2/founding_residual.py
"""
import json
import re
from datetime import date

import pandas as pd

TODAY = date(2026, 6, 10)
ROOT = "c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons"
PARQUET = f"{ROOT}/redrob-ranker/data/candidates_flat.parquet"
JSONL = (f"{ROOT}/hack2skill/The Data & AI Challenge/"
         "[PUB] India_runs_data_and_ai_challenge/"
         "India_runs_data_and_ai_challenge/candidates.jsonl")
DUMP = f"{ROOT}/redrob-ranker/eda/round2/founding_suspect_dumps.txt"

YOUNG = {"sarvam ai": 2023, "krutrim": 2023, "glance": 2019, "rephrase.ai": 2019,
         "cred": 2018, "observe.ai": 2017, "niramai": 2017, "saarthi.ai": 2017,
         "yellow.ai": 2016, "verloop.io": 2016}
SUSPECTS = ["CAND_0011687", "CAND_0017960", "CAND_0011432", "CAND_0088385",
            "CAND_0005509", "CAND_0008049", "CAND_0024878", "CAND_0077296",
            "CAND_0085851", "CAND_0051615"]

CORE_AI_TITLES = {
    "ML Engineer", "Machine Learning Engineer", "Senior Machine Learning Engineer",
    "Staff Machine Learning Engineer", "Applied ML Engineer", "Junior ML Engineer",
    "AI Engineer", "Senior AI Engineer", "Lead AI Engineer", "AI Specialist",
    "AI Research Engineer", "Senior Applied Scientist", "Senior Software Engineer (ML)",
    "Data Scientist", "Senior Data Scientist", "NLP Engineer", "Senior NLP Engineer",
    "Search Engineer", "Recommendation Systems Engineer", "Computer Vision Engineer",
}
YOE_PAT = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)(?:'| of)?\s*(?:hands-on\s+)?experience")


def parse_date(s):
    if not s:
        return None
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def main():
    df = pd.read_parquet(PARQUET)
    df["ai_titled"] = df.current_title.isin(CORE_AI_TITLES)
    ai_ids = set(df.loc[df.ai_titled, "candidate_id"])

    tenures = []   # every tenure at a YOUNG company, pool-wide
    suspects_raw = {}
    awscert = 0
    awscert_years = {}
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cid = r["candidate_id"]
            if cid in SUSPECTS:
                suspects_raw[cid] = r
            for c in (r.get("certifications") or []):
                if "AWS Certified Machine Learning" in (c.get("name") or ""):
                    awscert += 1
                    awscert_years[c.get("year")] = awscert_years.get(c.get("year"), 0) + 1
            for j in (r.get("career_history") or []):
                comp = (j.get("company") or "").strip().lower()
                fy = YOUNG.get(comp)
                if fy is None:
                    continue
                sd = parse_date(j.get("start_date"))
                ed = parse_date(j.get("end_date")) if j.get("end_date") else TODAY
                tenures.append(dict(
                    candidate_id=cid, company=comp, founded=fy,
                    start_year=sd.year if sd else None,
                    years_early=(fy - sd.year) if sd else None,
                    dur=j.get("duration_months"),
                    age_at_today=(TODAY.year - fy) * 12,
                    ai=cid in ai_ids))
    t = pd.DataFrame(tenures)
    print(f"tenures at 10 young companies pool-wide: {len(t)} "
          f"({t.candidate_id.nunique()} candidates)")

    print("\n--- 1. start-year distribution per company (founding-aware generator?) ---")
    for comp in ["sarvam ai", "krutrim", "cred", "glance"]:
        sub = t[t.company == comp]
        vc = sub.start_year.value_counts().sort_index()
        pre = int((sub.years_early > 0).sum())
        print(f"{comp} (founded {YOUNG[comp]}): n={len(sub)}, pre-founding {pre} "
              f"({100*pre/len(sub):.1f}%)")
        print("   " + " ".join(f"{y}:{c}" for y, c in vc.items()))

    print("\n--- 2. violation rate conditional on tenure: AI vs non-AI / grade ---")
    g = t.groupby("ai").agg(n=("candidate_id", "nunique"),
                            viol=("years_early", lambda s: (s > 0).sum()))
    g["rate"] = (g.viol / g.n).round(3)
    print(g.to_string())
    # within AI pool by grade>=4 (need grades — quick prefix match on P-grade 4/5 ids
    # via job_descriptions strong substrings is overkill; merge from crosscheck logic)
    pref = {  # grade>=4 paragraph prefixes (48 chars) from PARA_GRADES
        "Built recommendation-style features at a mid-sta": 4,
        "Owned the ranking layer for an e-commerce search": 5,
        "Trained and shipped multiple ranking models for ": 5,
        "Developed a semantic search feature for an inter": 5,
        "Implemented a RAG-based customer support chatbot": 4,
        "Built a content recommendation system serving 10": 5,
        "Fine-tuned LLaMA-2-7B and Mistral-7B variants us": 5,
        "Built a RAG-based ranking pipeline serving 50M+ ": 5,
        "Built and shipped a production recommendation sy": 5,
        "Owned the end-to-end ranking pipeline at a recom": 5,
        "Owned the design and rollout of a large-scale se": 5,
        "Led the migration from keyword-based to embeddin": 5,
        "Built systems that understand what users are loo": 5,
        "Shipped the personalization infrastructure: the ": 5,
        "Designed the ranking layer for the company's fla": 5,
        "Owned the search and discovery experience end-to": 5,
        "Led the engineering team building infrastructure": 5,
    }
    def g45(text):
        return any(p.strip()[:48] in pref for p in (text or "").split("\n"))
    df["grade45"] = [g45(x) for x in df.job_descriptions.fillna("")]
    ai45 = set(df.loc[df.ai_titled & df.grade45, "candidate_id"])
    tai = t[t.ai].copy()
    tai["g45"] = tai.candidate_id.isin(ai45)
    g2 = tai.groupby("g45").agg(n=("candidate_id", "nunique"),
                                viol=("years_early", lambda s: (s > 0).sum()))
    g2["rate"] = (g2.viol / g2.n).round(3)
    print("\nAI-pool tenures at young cos, by grade>=4:")
    print(g2.to_string())

    print("\n--- 3. README-literal: tenure duration > company age + 12mo slack ---")
    t["over_age"] = t.dur > (t.age_at_today + 12)
    o = t[t.over_age]
    print(f"tenure-longer-than-company-age violators: {len(o)} rows, "
          f"{o.candidate_id.nunique()} candidates")
    if len(o):
        print(o.groupby("company").size().to_string())
        odf = o.merge(df[["candidate_id", "current_title", "ai_titled", "grade45",
                          "yoe", "country", "sig_response_rate", "days_since_active"]],
                      on="candidate_id")
        print(odf.sort_values(["ai_titled", "grade45", "dur"], ascending=False)
              .head(40).to_string(index=False))
        print(f"AI-titled among them: {int(odf.ai_titled.sum())}, "
              f"AI+grade45: {int((odf.ai_titled & odf.grade45).sum())}")
        print("duration-over-age (months) distribution:")
        print((o.dur - o.age_at_today).describe().to_string())

    print("\n--- 4. internal consistency of 10 worst worthy suspects ---")
    dump = open(DUMP, "w", encoding="utf-8")
    for cid in SUSPECTS:
        r = suspects_raw[cid]
        prof = r["profile"]
        jobs = r.get("career_history") or []
        bad = []
        for j in jobs:
            sd, ed = parse_date(j.get("start_date")), parse_date(j.get("end_date"))
            dur = j.get("duration_months")
            if sd and dur is not None:
                eff = TODAY if (j.get("is_current") or ed is None) else min(ed, TODAY)
                if abs(dur - (eff - sd).days / 30.4375) > 3:
                    bad.append("B")
        starts = [parse_date(j["start_date"]) for j in jobs if j.get("start_date")]
        ends = [TODAY if (j.get("is_current") or not j.get("end_date"))
                else min(parse_date(j["end_date"]), TODAY) for j in jobs]
        yoe = prof.get("years_of_experience")
        if starts and yoe * 12 > (max(ends) - min(starts)).days / 30.4375 + 24:
            bad.append("C")
        m = YOE_PAT.search(prof.get("summary") or "")
        if m and abs(float(m.group(1)) - yoe) > 1.5:
            bad.append("D")
        if any(s.get("proficiency") == "expert" and s.get("duration_months") == 0
               for s in (r.get("skills") or [])):
            bad.append("A")
        if any((c.get("year") or 0) > 2026 for c in (r.get("certifications") or [])):
            bad.append("E")
        yc = [(j.get("company"), j.get("start_date"), j.get("end_date"),
               j.get("duration_months")) for j in jobs
              if (j.get("company") or "").strip().lower() in YOUNG]
        print(f"{cid}: hard markers {sorted(set(bad)) or 'NONE'}; yoe {yoe}; "
              f"summary-yoe {m.group(1) if m else 'n/a'}; young-co jobs: {yc}")
        dump.write(f"\n{'='*90}\nSUSPECT {cid}\n{'='*90}\n")
        dump.write(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    dump.close()

    print("\n--- 5. AWS ML cert prevalence (E-family context) ---")
    print(f"pool-wide 'AWS Certified Machine Learning*' certs: {awscert}")
    print(f"year distribution: {dict(sorted(awscert_years.items()))}")
    print(f"\nsuspect dumps -> {DUMP}")


if __name__ == "__main__":
    main()
