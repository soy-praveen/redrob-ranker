# -*- coding: utf-8 -*-
"""
HONEYPOT CROSS-CHECKER (round 2, DQ protection).

T1  93 honeypots x relevance pool (grade-4/5 paragraphs / AI titles); raw-JSONL
    re-verification of every overlap.
T2  14 DQ-critical AI-bait ids: re-verify each trips >=1 hard marker from RAW.
T3  Founding-date residual scan WITHIN the AI-titled pool (1,179) only.
T4  Characterize the 13 'extra' honeypots (93 found vs ~80 stated).
T5  Write eda/round2/honeypot_final.csv (candidate_id, markers, ai_titled).

Run: PYTHONIOENCODING=utf-8 python eda/round2/honeypot_crosscheck.py
"""
import json
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
HP_CSV = f"{ROOT}/redrob-ranker/eda/honeypot_candidates.csv"
OUT_CSV = f"{ROOT}/redrob-ranker/eda/round2/honeypot_final.csv"
DUMP = f"{ROOT}/redrob-ranker/eda/round2/honeypot_raw_dumps.txt"

# ---- hand grades from eda/tier5-language.py::PARA_GRADES (60-char prefixes) ----
PARA_GRADES = {
    "Enterprise sales of cloud software solutions into the mid-m": ("P00", 0),
    "Customer support team lead at a SaaS product. Managed a tea": ("P01", 0),
    "Marketing leadership role at a B2B SaaS company. Owned the ": ("P02", 0),
    "Business analyst at a consulting firm, working primarily wi": ("P03", 0),
    "Brand design and creative direction at a consumer-products ": ("P04", 0),
    "Mechanical engineering design role at a hardware-product co": ("P05", 0),
    "Senior accounting role at a mid-sized company — month-end c": ("P06", 0),
    "Content writing and SEO strategy for a tech-focused publica": ("P07", 0),
    "Operations management role at a logistics company. Owned da": ("P08", 0),
    "Cloud infrastructure and DevOps work at an enterprise SaaS ": ("P09", 0),
    "Android mobile development using Java and (more recently) K": ("P10", 0),
    "Frontend engineering at a media company. React, TypeScript,": ("P11", 0),
    "Java backend development at a large enterprise — Spring Boo": ("P12", 0),
    "Full-stack web application development at a SaaS company. B": ("P13", 0),
    "Test automation and QA engineering for a fintech product. B": ("P14", 0),
    "Designed and maintained the analytical data warehouse on Sn": ("P15", 1),
    "Built and maintained data pipelines on Apache Airflow proce": ("P16", 1),
    "Backend + data hybrid role at a growth-stage startup. Built": ("P17", 1),
    "Implemented streaming data pipelines on Kafka and Spark Str": ("P18", 1),
    "Mixed data science and analytics-engineering role at a mark": ("P19", 2),
    "Backend development with Python (FastAPI), PostgreSQL, and ": ("P20", 2),
    "Contributed to ML feature engineering and model deployment ": ("P21", 3),
    "Built recommendation-style features at a mid-stage startup ": ("P22", 4),
    "Built computer vision models for our product's image modera": ("P23", 2),
    "Worked on time-series forecasting models for supply-chain d": ("P24", 2),
    "Worked on customer-facing predictive modeling for an e-comm": ("P25", 3),
    "Built NLP pipelines for sentiment analysis and document cla": ("P26", 3),
    "Owned the ranking layer for an e-commerce search product, e": ("P27", 5),
    "Trained and shipped multiple ranking models for our product": ("P28", 5),
    "Developed a semantic search feature for an internal knowled": ("P29", 5),
    "Implemented a RAG-based customer support chatbot integrated": ("P30", 4),
    "Built a content recommendation system serving 10M+ users th": ("P31", 5),
    "Built and operated production ML pipelines using MLflow for": ("P32", 3),
    "Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA an": ("P33", 5),
    "Built a RAG-based ranking pipeline serving 50M+ queries per": ("P34", 5),
    "Built and shipped a production recommendation system at a m": ("P35", 5),
    "Owned the end-to-end ranking pipeline at a recommendations-": ("P36", 5),
    "Owned the design and rollout of a large-scale semantic sear": ("P37", 5),
    "Led the migration from keyword-based to embedding-based sea": ("P38", 5),
    "Built systems that understand what users are looking for an": ("P39", 5),
    "Shipped the personalization infrastructure: the system that": ("P40", 5),
    "Designed the ranking layer for the company's flagship produ": ("P41", 5),
    "Owned the search and discovery experience end-to-end at a c": ("P42", 5),
    "Led the engineering team building infrastructure to surface": ("P43", 5),
}

CORE_AI_TITLES = {
    "ML Engineer", "Machine Learning Engineer", "Senior Machine Learning Engineer",
    "Staff Machine Learning Engineer", "Applied ML Engineer", "Junior ML Engineer",
    "AI Engineer", "Senior AI Engineer", "Lead AI Engineer", "AI Specialist",
    "AI Research Engineer", "Senior Applied Scientist",
    "Senior Software Engineer (ML)",
    "Data Scientist", "Senior Data Scientist",
    "NLP Engineer", "Senior NLP Engineer",
    "Search Engineer", "Recommendation Systems Engineer",
}
CV_ONLY_TITLES = {"Computer Vision Engineer"}

# Founding years: honeypots.py map + prompt-listed young companies
FOUNDING = {
    "sarvam ai": 2023, "krutrim": 2023, "glance": 2019, "rephrase.ai": 2019,
    "cred": 2018, "observe.ai": 2017, "niramai": 2017, "saarthi.ai": 2017,
    "yellow.ai": 2016, "verloop.io": 2016, "meesho": 2015, "unacademy": 2015,
    "pharmeasy": 2015, "upgrad": 2015, "phonepe": 2015, "wysa": 2015,
    "locobuzz": 2015, "swiggy": 2014, "razorpay": 2014, "haptik": 2013,
    "mad street den": 2013, "nykaa": 2012, "byju's": 2011, "vedantu": 2011,
    "paytm": 2010, "ola": 2010, "freshworks": 2010, "zomato": 2008,
    "dream11": 2008, "policybazaar": 2008, "aganitha": 2008, "inmobi": 2007,
    "flipkart": 2007,
    # prompt-listed (apply only if company appears in data)
    "anthropic": 2021, "mistral": 2023, "mistral ai": 2023, "xai": 2023,
    "perplexity": 2022, "perplexity ai": 2022, "zepto": 2021, "jupiter": 2019,
    "fi": 2019, "fi money": 2019, "bharatpe": 2018, "openai": 2015,
    "hugging face": 2016, "scale ai": 2016, "cohere": 2019, "deepmind": 2010,
    "groww": 2016, "udaan": 2016, "sharechat": 2015, "cure.fit": 2016,
    "slice": 2016, "jar": 2021, "khatabook": 2018,
}

YOE_PAT = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)(?:'| of)?\s*(?:hands-on\s+)?experience")


def parse_date(s):
    if not s:
        return None
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def months_between(a, b):
    return (b - a).days / 30.4375


def verify_raw(r):
    """Recompute the 5 hard markers from a raw JSONL record only. Returns
    dict marker -> evidence string (only markers that trip)."""
    out = {}
    prof = r.get("profile", {})
    jobs = r.get("career_history") or []
    # A: expert proficiency with 0 months
    a = [s for s in (r.get("skills") or [])
         if s.get("proficiency") == "expert" and s.get("duration_months") == 0]
    if a:
        out["A"] = "; ".join(f"{s['name']}: expert, 0mo, {s.get('endorsements')} endorsements"
                             for s in a)
    # B: |duration_months - months(start, end)| > 3
    bev = []
    for j in jobs:
        sd, ed, dur = parse_date(j.get("start_date")), parse_date(j.get("end_date")), j.get("duration_months")
        if sd and dur is not None:
            eff = TODAY if (j.get("is_current") or ed is None) else min(ed, TODAY)
            gap = dur - months_between(sd, eff)
            if abs(gap) > 3:
                bev.append(f"{j.get('company')}/{j.get('title')}: claims {dur}mo but "
                           f"{j.get('start_date')}->{j.get('end_date') or 'current'} = "
                           f"{months_between(sd, eff):.1f}mo (gap {gap:+.1f}mo)")
    if bev:
        out["B"] = "; ".join(bev)
    # C: yoe*12 > span+24
    starts = [parse_date(j.get("start_date")) for j in jobs if j.get("start_date")]
    ends = [TODAY if (j.get("is_current") or not j.get("end_date"))
            else min(parse_date(j["end_date"]), TODAY) for j in jobs]
    yoe = prof.get("years_of_experience")
    if starts and yoe is not None:
        span = months_between(min(starts), max(ends))
        if yoe * 12 > span + 24:
            out["C"] = (f"yoe field {yoe}y = {yoe*12:.0f}mo but whole career "
                        f"{min(starts)} -> {max(ends)} spans {span:.0f}mo "
                        f"(excess {yoe*12-span:.0f}mo)")
    # D: summary-text yoe vs yoe field gap > 1.5
    m = YOE_PAT.search(prof.get("summary") or "")
    if m and yoe is not None and abs(float(m.group(1)) - yoe) > 1.5:
        out["D"] = (f"summary says '{m.group(0)}' ({m.group(1)}y) but yoe field = {yoe}y "
                    f"(gap {abs(float(m.group(1))-yoe):.1f}y)")
    # E: cert year > 2026
    e = [c for c in (r.get("certifications") or []) if (c.get("year") or 0) > 2026]
    if e:
        out["E"] = "; ".join(f"{c.get('name')} dated {c.get('year')}" for c in e)
    return out


def main():
    df = pd.read_parquet(PARQUET)
    hp = pd.read_csv(HP_CSV)
    hp_ids = set(hp.candidate_id)
    print(f"parquet {df.shape}; honeypot CSV rows {len(hp)} (unique {len(hp_ids)})")

    # paragraph grades per candidate
    pref48 = {k[:48]: v for k, v in PARA_GRADES.items()}
    def grade_of(text):
        best, pids = -1, []
        for p in (text or "").split("\n"):
            p = p.strip()
            if not p:
                continue
            v = pref48.get(p[:48])
            if v:
                pids.append(f"{v[0]}:{v[1]}")
                best = max(best, v[1])
        return best, ",".join(pids)
    g = [grade_of(t) for t in df.job_descriptions.fillna("")]
    df["best_grade"] = [x[0] for x in g]
    df["para_ids"] = [x[1] for x in g]
    df["ai_titled"] = df.current_title.isin(CORE_AI_TITLES | CV_ONLY_TITLES)
    print(f"AI/CV-titled pool: {int(df.ai_titled.sum())}")
    print("best_grade distribution (pool):")
    print(df.best_grade.value_counts().sort_index().to_string())

    # ---------- raw JSONL pass: collect honeypot records + AI-pool careers ----
    ai_ids = set(df.loc[df.ai_titled, "candidate_id"])
    raw_hp = {}
    ai_careers = {}
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            cid = line[line.find("CAND_"):line.find("CAND_") + 12]
            if cid in hp_ids:
                raw_hp[cid] = json.loads(line)
            elif cid in ai_ids:
                r = json.loads(line)
                ai_careers[cid] = r.get("career_history") or []
    print(f"raw records collected: honeypots {len(raw_hp)}, AI careers {len(ai_careers)}")

    # ================= T1: overlap with relevance pool =================
    hpx = hp.merge(df[["candidate_id", "best_grade", "para_ids", "ai_titled",
                       "location", "country", "sig_response_rate",
                       "days_since_active", "sig_willing_to_relocate",
                       "sig_notice_period_days"]], on="candidate_id")
    n45 = int((hpx.best_grade >= 4).sum())
    nai = int(hpx.ai_titled.sum())
    print(f"\n=== T1: of the 93 honeypots: grade>=4 paragraphs: {n45}; "
          f"AI/CV-titled: {nai}; grade>=4 AND AI-titled: "
          f"{int(((hpx.best_grade>=4)&hpx.ai_titled).sum())} ===")
    overlap = hpx[(hpx.best_grade >= 4) | hpx.ai_titled].sort_values("candidate_id")
    print(overlap[["candidate_id", "families", "best_grade", "para_ids",
                   "current_title", "current_company", "yoe",
                   "summary_yoe"]].to_string(index=False))

    dump = open(DUMP, "w", encoding="utf-8")
    print("\n--- raw-JSONL re-verification of each overlap ---")
    t1_fail = []
    for cid in overlap.candidate_id:
        ev = verify_raw(raw_hp[cid])
        csv_fams = set(x.split("_")[0] for x in
                       hp.loc[hp.candidate_id == cid, "families"].iloc[0].split(","))
        ok = bool(ev)
        match = set(ev) == csv_fams
        print(f"{cid}: raw markers {sorted(ev)} (csv said {sorted(csv_fams)}) "
              f"{'OK' if ok else '** NO MARKER FROM RAW **'}"
              f"{'' if match else ' [family set differs]'}")
        for k, v in sorted(ev.items()):
            print(f"    [{k}] {v}")
        if not ok:
            t1_fail.append(cid)
        dump.write(f"\n{'='*90}\n{cid}  families={sorted(ev)}\n{'='*90}\n")
        dump.write(json.dumps(raw_hp[cid], indent=1, ensure_ascii=False) + "\n")
    print(f"T1 verdict: {len(overlap)} overlaps, {len(overlap)-len(t1_fail)} confirmed "
          f"from raw; failures: {t1_fail or 'NONE'}")

    # ================= T2: the 14 DQ-critical AI-bait ids =================
    DQ14 = ["CAND_0001610", "CAND_0010770", "CAND_0013536", "CAND_0019480",
            "CAND_0022870", "CAND_0037000", "CAND_0039521", "CAND_0039754",
            "CAND_0055992", "CAND_0071115", "CAND_0091534", "CAND_0093331",
            "CAND_0093547", "CAND_0095619"]
    print("\n=== T2: 14 DQ-critical AI-bait honeypots, raw re-verification ===")
    t2_fail = []
    for cid in DQ14:
        if cid not in raw_hp:
            print(f"{cid}: ** NOT IN 93-id SET **"); t2_fail.append(cid); continue
        ev = verify_raw(raw_hp[cid])
        row = df.loc[df.candidate_id == cid].iloc[0]
        print(f"{cid}: markers {sorted(ev)} | {row.current_title} @ {row.current_company} "
              f"| grade {row.best_grade} | {'OK' if ev else '** NO MARKER **'}")
        for k, v in sorted(ev.items()):
            print(f"    [{k}] {v}")
        if not ev:
            t2_fail.append(cid)
    print(f"T2 verdict: 14/14 trip >=1 hard marker: {not t2_fail} "
          f"(failures: {t2_fail or 'NONE'})")

    # ================= T3: founding-date scan within AI pool =================
    print("\n=== T3: founding-date residual scan (AI/CV-titled pool only) ===")
    comp_counts = {}
    for jobs in ai_careers.values():
        for j in jobs:
            c = (j.get("company") or "").strip()
            comp_counts[c] = comp_counts.get(c, 0) + 1
    known = {c: FOUNDING[c.lower()] for c in comp_counts if c.lower() in FOUNDING}
    unknown = sorted(c for c in comp_counts if c.lower() not in FOUNDING)
    print(f"companies in AI-pool careers: {len(comp_counts)}; with founding year: "
          f"{len(known)}; without: {unknown}")
    viol = []
    for cid, jobs in ai_careers.items():
        for j in jobs:
            fy = FOUNDING.get((j.get("company") or "").strip().lower())
            sd = parse_date(j.get("start_date"))
            if fy and sd and sd.year < fy:
                viol.append(dict(candidate_id=cid, company=j.get("company"),
                                 founded=fy, start=j.get("start_date"),
                                 years_early=fy - sd.year))
    vdf = pd.DataFrame(viol)
    print(f"AI-pool founding violations: {len(vdf)} job-rows, "
          f"{vdf.candidate_id.nunique() if len(vdf) else 0} candidates")
    if len(vdf):
        print(vdf.company.value_counts().to_string())
        print("years_early distribution:")
        print(vdf.years_early.value_counts().sort_index().to_string())
        vsum = (vdf.groupby("candidate_id")
                .agg(worst_early=("years_early", "max"),
                     companies=("company", lambda s: ",".join(sorted(set(s)))))
                .reset_index()
                .merge(df[["candidate_id", "best_grade", "current_title", "yoe",
                           "location", "country", "sig_willing_to_relocate",
                           "sig_response_rate", "days_since_active",
                           "sig_notice_period_days"]], on="candidate_id"))
        vsum["already_hp"] = vsum.candidate_id.isin(hp_ids)
        vsum["top100_worthy"] = ((vsum.best_grade >= 4) & (vsum.yoe.between(4, 10)) &
                                 ((vsum.country == "India") | vsum.sig_willing_to_relocate) &
                                 (vsum.sig_response_rate >= 0.2) &
                                 (vsum.days_since_active <= 100))
        print("\nviolators with grade>=4 (could enter top-100 superset):")
        sub = vsum[vsum.best_grade >= 4].sort_values(
            ["top100_worthy", "best_grade", "worst_early"], ascending=False)
        print(sub.to_string(index=False) if len(sub) else "  NONE")
        print(f"\ntop100_worthy (full gate) count: {int(vsum.top100_worthy.sum())}")
        worthy = vsum[vsum.top100_worthy]
        for cid in worthy.candidate_id:
            r = json.loads([l for l in open(JSONL, encoding="utf-8")
                            if cid in l[:40]][0]) if cid not in raw_hp else raw_hp[cid]
            dump.write(f"\n{'='*90}\nT3 WORTHY FOUNDING-VIOLATOR {cid}\n{'='*90}\n")
            dump.write(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
        # magnitude framing: worthy violators vs pool-wide noise
        print("\nworst_early by grade band (AI pool violators):")
        print(vsum.groupby(vsum.best_grade >= 4).worst_early.describe().to_string())

    # ================= T4: characterize the 13 extras =================
    print("\n=== T4: 93 vs ~80 — family pattern census and noise audit ===")
    print("exact family-combination counts:")
    print(hp.families.value_counts().to_string())
    # magnitude/separation per family member
    hpm = hp.copy()
    hpm["D_gap"] = (hpm.summary_yoe - hpm.yoe).abs().round(1)
    hpm["C_excess_mo"] = (hpm.yoe * 12 - hpm.career_span_months).round(0)
    print("\nper-family magnitude (is anyone near the threshold = plausible noise?):")
    for fam, col, thr in [("A_expert_zero_months", "n_expert_zero_months", 1),
                          ("B_date_mismatch", "max_date_mismatch", 3),
                          ("C_yoe_vs_span", "C_excess_mo", 24),
                          ("D_summary_yoe_gap", "D_gap", 1.5)]:
        sub = hpm[hpm.families.str.contains(fam)]
        print(f"  {fam}: n={len(sub)}, {col} range "
              f"{sub[col].min()}..{sub[col].max()}, threshold {thr}; "
              f"5 smallest: {sorted(sub[col])[:5]}")
    e = hpm[hpm.families.str.contains("E_cert_future")]
    print(f"  E_cert_future: n={len(e)}, all cert_max_year values: "
          f"{sorted(e.cert_max_year.unique())}")
    # E signature: same cert name?
    enames = {}
    for cid in e.candidate_id:
        for c in (raw_hp[cid].get("certifications") or []):
            if (c.get("year") or 0) > 2026:
                enames[f"{c['name']}|{c['year']}"] = enames.get(f"{c['name']}|{c['year']}", 0) + 1
    print(f"  E future-cert name|year signature: {enames}")
    # A signature
    a = hpm[hpm.families.str.contains("A_expert")]
    print(f"  A n_expert_zero_months values: {sorted(a.n_expert_zero_months.unique())}")
    # population near-threshold checks (how island-like is each boundary?)
    df2 = df.copy()
    df2["summary_yoe"] = [float(m.group(1)) if (m := YOE_PAT.search(s)) else np.nan
                          for s in df2.summary.fillna("")]
    dgap = (df2.summary_yoe - df2.yoe).abs()
    print("\npopulation near-threshold counts (noise leakage into each marker):")
    print(f"  D gap in (0.7,1.5]: {int(((dgap>0.7)&(dgap<=1.5)).sum())}, "
          f"(1.5,3.0]: {int(((dgap>1.5)&(dgap<=3.0)).sum())}, >1.5 total {int((dgap>1.5).sum())}")
    cex = df2.yoe * 12 - df2.career_span_months
    print(f"  C excess in (0,24]: {int(((cex>0)&(cex<=24)).sum())}, "
          f"(24,96): {int(((cex>24)&(cex<96)).sum())}, >=96: {int((cex>=96).sum())}")
    print(f"  B mismatch in (0.8,3]: {int(((df2.date_duration_mismatch>0.8)&(df2.date_duration_mismatch<=3)).sum())} "
          f"(parquet col), >3: {int((df2.date_duration_mismatch>3).sum())}")
    print(f"  A n_expert_zero_months==1or2 pool-wide: "
          f"{int(df2.n_expert_zero_months.isin([1,2]).sum())}, >=3: {int((df2.n_expert_zero_months>=3).sum())}")

    # tamper-action view: BCD/CD/BC = one yoe-tamper planting each
    combo = hp.families.value_counts()
    singles = {f: int(combo.get(f, 0)) for f in
               ["A_expert_zero_months", "B_date_mismatch", "C_yoe_vs_span",
                "D_summary_yoe_gap", "E_cert_future"]}
    multi = int(len(hp) - sum(singles.values()))
    print(f"\ntamper-action census: singles {singles}, multi-marker profiles {multi}, "
          f"total {len(hp)}")

    # ================= T5: final exclusion CSV =================
    out = hpx[["candidate_id", "families", "ai_titled"]].rename(
        columns={"families": "markers"}).sort_values("candidate_id")
    out.to_csv(OUT_CSV, index=False)
    print(f"\n=== T5: wrote {len(out)} rows -> {OUT_CSV} ===")
    print(f"ai_titled among them: {int(out.ai_titled.sum())}")
    # cost-of-exclusion audit: what do we lose if some of 93 are not honeypots?
    print("\ncost audit: 93 honeypots by best_grade:")
    print(hpx.best_grade.value_counts().sort_index().to_string())
    dump.close()
    print(f"raw dumps -> {DUMP}")


if __name__ == "__main__":
    main()
