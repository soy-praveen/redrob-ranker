# -*- coding: utf-8 -*-
"""
EXEMPLAR DEEP-READER — Redrob Senior AI Engineer JD vs 100K synthetic candidates.

Steps:
  1. Build "plausible elite pool" from the parquet:
     ML/AI/search/retrieval signal in title/headline/summary, yoe 4-11,
     India OR willing_to_relocate, days_since_active <= 90.
  2. Heuristic-rank the pool (AI/retrieval relevance + recency + response rate),
     pull full raw JSONL records for the top N for manual deep-reading.
  3. Also pull "embedding-bait" records: high keyword match but recruiter red flags
     (consulting-only careers, long notice + inactive, etc.).
  4. Salary stats for the elite pool.

Run:  PYTHONIOENCODING=utf-8 python eda/exemplars.py [--dump-records]
"""
import json
import re
import sys

import numpy as np
import pandas as pd

PARQUET = "c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons/redrob-ranker/data/candidates_flat.parquet"
JSONL = ("c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons/hack2skill/"
         "The Data & AI Challenge/[PUB] India_runs_data_and_ai_challenge/"
         "India_runs_data_and_ai_challenge/candidates.jsonl")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 100)

df = pd.read_parquet(PARQUET)
print(f"Loaded parquet: {df.shape}")

# ---------------------------------------------------------------- text fields
for c in ["headline", "summary", "current_title", "job_titles", "job_descriptions",
          "skill_names", "job_companies", "job_industries", "location", "country"]:
    df[c] = df[c].fillna("")

df["text_title"] = (df["current_title"] + " | " + df["headline"]).str.lower()
df["text_all"] = (df["current_title"] + " | " + df["headline"] + " | " + df["summary"]
                  + " | " + df["job_titles"]).str.lower()
df["text_jobs"] = df["job_descriptions"].str.lower()
df["text_skills"] = df["skill_names"].str.lower()

# ------------------------------------------------------- 1. ELITE POOL FILTER
AI_TITLE_RE = re.compile(
    r"machine learning|ml engineer|ai engineer|applied scientist|applied ai|"
    r"\bnlp\b|deep learning|data scientist|search engineer|recommendation|"
    r"information retrieval|\bllm\b|genai|generative ai|mlops|relevance|ranking",
    re.I)
RETRIEVAL_RE = re.compile(
    r"embedding|vector|retriev|semantic search|\brag\b|recommendation|ranking|"
    r"\bndcg\b|\bmrr\b|faiss|pinecone|weaviate|qdrant|milvus|elasticsearch|"
    r"opensearch|\bbm25\b|two[- ]tower|learning[- ]to[- ]rank|hybrid search|re[- ]?rank",
    re.I)

has_ai_title = df["text_title"].str.contains(AI_TITLE_RE) | df["text_all"].str.contains(AI_TITLE_RE)
has_retrieval_any = (df["text_all"].str.contains(RETRIEVAL_RE)
                     | df["text_jobs"].str.contains(RETRIEVAL_RE))
ml_ai_signal = has_ai_title | has_retrieval_any

yoe_ok = df["yoe"].between(4, 11)
india = df["country"].str.lower().eq("india")
loc_ok = india | df["sig_willing_to_relocate"]
active_ok = df["days_since_active"] <= 90

elite = df[ml_ai_signal & yoe_ok & loc_ok & active_ok].copy()
print("\n=== 1. ELITE POOL ===")
print(f"ML/AI/retrieval signal anywhere:       {int(ml_ai_signal.sum()):>6}")
print(f"  + yoe 4-11:                          {int((ml_ai_signal & yoe_ok).sum()):>6}")
print(f"  + India or willing_to_relocate:      {int((ml_ai_signal & yoe_ok & loc_ok).sum()):>6}")
print(f"  + active <=90d  => ELITE POOL:       {len(elite):>6}")
print(f"Elite pool in-India share: {elite['country'].str.lower().eq('india').mean():.1%}")

# --------------------------------------------------- 2. HEURISTIC DEEP SCORE
def count_hits(series, regex):
    return series.str.count(regex)

elite["kw_retrieval_jobs"] = count_hits(elite["text_jobs"], RETRIEVAL_RE.pattern)
elite["kw_retrieval_prof"] = count_hits(elite["text_all"], RETRIEVAL_RE.pattern)
EVAL_RE = (r"ndcg|\bmrr\b|recall@|precision@|a/b test|ab test|offline eval|"
           r"online metric|click[- ]through|ctr uplift|interleav")
elite["kw_eval"] = count_hits(elite["text_jobs"] + " " + elite["text_all"], EVAL_RE)
PROD_RE = (r"production|deployed|served|serving|latency|qps|scale[ds]?\b|"
           r"millions of|launched|shipped")
elite["kw_prod"] = count_hits(elite["text_jobs"], PROD_RE)

elite["h_score"] = (
    2.0 * elite["text_title"].str.contains(AI_TITLE_RE).astype(int)
    + 1.0 * np.minimum(elite["kw_retrieval_jobs"], 6)
    + 0.8 * np.minimum(elite["kw_eval"], 4)
    + 0.4 * np.minimum(elite["kw_prod"], 5)
    + 1.0 * (1 - elite["days_since_active"] / 90.0)
    + 1.0 * elite["sig_response_rate"].fillna(0)
)
elite = elite.sort_values("h_score", ascending=False)

cols = ["candidate_id", "name", "current_title", "current_company", "yoe", "location",
        "days_since_active", "sig_response_rate", "sig_notice_period_days",
        "sig_salary_min", "sig_salary_max", "kw_retrieval_jobs", "kw_eval", "h_score"]
print("\n=== TOP 40 BY HEURISTIC (deep-read shortlist source) ===")
print(elite[cols].head(40).to_string(index=False))

# ------------------------------------------- 3. EMBEDDING-BAIT / TRAP FINDER
CONSULT_RE = re.compile(r"\b(tcs|tata consultancy|infosys|wipro|accenture|cognizant|"
                        r"capgemini|hcl|tech mahindra|mindtree|lti|mphasis)\b", re.I)

def consulting_only(row):
    comps = [c.strip() for c in row.split(" | ") if c.strip()]
    if not comps:
        return False
    return all(CONSULT_RE.search(c) for c in comps)

df["consulting_only"] = df["job_companies"].apply(consulting_only)
kw_rich = df["text_skills"].str.count(
    r"rag|langchain|vector|embedding|pinecone|llm|gpt|hugging ?face|transformer") >= 3

bait1 = df[df["consulting_only"] & kw_rich & df["yoe"].between(4, 11)]
print(f"\nConsulting-ONLY careers in dataset: {int(df['consulting_only'].sum())}")
print(f"Consulting-only + AI-keyword-rich skills + yoe 4-11 (bait type A): {len(bait1)}")

bait2 = df[(df["text_all"].str.contains(RETRIEVAL_RE)) & df["yoe"].between(5, 9)
           & ((df["sig_notice_period_days"] >= 90)
              | (df["days_since_active"] > 150)
              | (df["sig_response_rate"] < 0.10))]
print(f"Strong retrieval keywords BUT (notice>=90d OR inactive>150d OR resp<10%) (bait type B): {len(bait2)}")

kw_stuffer = df[(~df["text_title"].str.contains(AI_TITLE_RE))
                & df["text_title"].str.contains(
                    r"marketing|sales|hr |recruit|account|finance|designer|support", regex=True)
                & kw_rich]
print(f"Non-tech title + AI-stuffed skills (keyword stuffers): {len(kw_stuffer)}")

# ------------------------------------------------------------ 4. SALARY STATS
print("\n=== 4. SALARY EXPECTATIONS (sig_salary_min/max), ELITE POOL ===")
sal = elite[elite["country"].str.lower().eq("india")][["sig_salary_min", "sig_salary_max"]].dropna()
print(f"India elite n={len(sal)}")
print(sal.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
top50 = elite.head(50)
sal50 = top50[["sig_salary_min", "sig_salary_max"]].dropna()
print(f"\nTop-50 heuristic elite salary (n={len(sal50)}): "
      f"min median={sal50['sig_salary_min'].median():.0f}, "
      f"max median={sal50['sig_salary_max'].median():.0f}, "
      f"max p90={sal50['sig_salary_max'].quantile(.9):.0f}")
allind = df[df['country'].str.lower().eq('india')][["sig_salary_min", "sig_salary_max"]].dropna()
print(f"All-India baseline (n={len(allind)}): min median={allind['sig_salary_min'].median():.0f}, "
      f"max median={allind['sig_salary_max'].median():.0f}")

# notice period distribution in elite
print("\nElite notice period days distribution:")
print(elite["sig_notice_period_days"].value_counts().sort_index().to_string())

# --------------------------------------------- 5. RECORD DUMP FOR DEEP READS
DEEP_READ_IDS = list(elite["candidate_id"].head(18))
# embedding-bait picks: top keyword-rich consulting-only + worst-logistics retrieval people
bait1s = bait1.assign(k=bait1["text_skills"].str.count(r"rag|vector|embedding|llm|pinecone")) \
              .sort_values("k", ascending=False)
bait2s = bait2.assign(k=bait2["text_jobs"].str.count(RETRIEVAL_RE.pattern)).sort_values("k", ascending=False)
BAIT_IDS = list(bait1s["candidate_id"].head(3)) + list(bait2s["candidate_id"].head(4))
BAIT_IDS = [i for i in BAIT_IDS if i not in DEEP_READ_IDS][:6]

print("\nDeep-read IDs:", DEEP_READ_IDS)
print("Bait IDs:", BAIT_IDS)

# ------------------------------- 6. TEMPLATE & CONSISTENCY QUANTIFICATION
print("\n=== 6. SUMMARY TEMPLATES (verbatim generator templates found in deep reads) ===")
T_ELITE = ("hands-on experience building production ML systems, "
           "with a focus on search, retrieval, and ranking")
T_SOLID = ("experience building ML-powered features in production. Strong background "
           "in NLP, recommendation systems")
T_STUFF = "excited about how AI and GenAI tools can augment my work"
df["summary"] = df["summary"].fillna("")
is_elite_t = df["summary"].str.contains(T_ELITE, regex=False)
is_solid_t = df["summary"].str.contains(T_SOLID, regex=False)
is_stuff_t = df["summary"].str.contains(T_STUFF, regex=False)
print(f"T_ELITE ('Senior AI engineer... search, retrieval, and ranking'): {int(is_elite_t.sum())}")
print(f"T_SOLID ('ML engineer... ML-powered features in production'):     {int(is_solid_t.sum())}")
print(f"T_STUFFER ('excited about how AI/GenAI tools can augment'):       {int(is_stuff_t.sum())}")
for name, mask in [("T_ELITE", is_elite_t), ("T_SOLID", is_solid_t), ("T_STUFF", is_stuff_t)]:
    sub = df[mask]
    print(f"  {name}: n={len(sub)} | india={sub['country'].str.lower().eq('india').mean():.0%} "
          f"| med days_since_active={sub['days_since_active'].median():.0f} "
          f"| med resp_rate={sub['sig_response_rate'].median():.2f} "
          f"| med notice={sub['sig_notice_period_days'].median():.0f} "
          f"| med sal_max={sub['sig_salary_max'].median():.1f} "
          f"| med yoe={sub['yoe'].median():.1f} "
          f"| open_to_work={sub['sig_open_to_work'].mean():.0%}")
elite_active = df[is_elite_t & (df['days_since_active'] <= 90)
                  & (df['country'].str.lower().eq('india') | df['sig_willing_to_relocate'])]
print(f"T_ELITE + active<=90d + (India|relocate): {len(elite_active)}")
print(f"  of those, notice<=60d: {int((elite_active['sig_notice_period_days']<=60).sum())}, "
      f"resp>=0.5: {int((elite_active['sig_response_rate']>=0.5).sum())}, "
      f"both: {int(((elite_active['sig_notice_period_days']<=60)&(elite_active['sig_response_rate']>=0.5)).sum())}")

print("\n=== 6b. SKILL-DURATION ANACHRONISMS (is tech-age violation a honeypot marker or noise?) ===")
# tech max plausible age (months) as of 2026-06: QLoRA(2023-05)=37, LangChain(2022-10)=44,
# LlamaIndex(2022-11)=43, RAG term(2020-05)=73, pgvector(2021-04)=62, Pinecone GA(2021-01)=65
TECH_MAX = {"QLoRA": 37, "LangChain": 44, "LlamaIndex": 43, "RAG": 73, "pgvector": 62, "Pinecone": 65}
viol_counts = {}
have_counts = {}
sj = df["skills_json"].fillna("[]")
for tech, mx in TECH_MAX.items():
    durs = sj.str.extract(r'"name": "' + tech + r'"[^}]*?"duration_months": (\d+)', expand=False)
    durs = pd.to_numeric(durs, errors="coerce")
    have = durs.notna()
    viol = durs > mx
    have_counts[tech] = int(have.sum())
    viol_counts[tech] = int(viol.sum())
    print(f"{tech:<11} have={have_counts[tech]:>6}  duration>{mx}mo: {viol_counts[tech]:>6} "
          f"({viol_counts[tech]/max(have_counts[tech],1):.0%} of holders)")

print("\n=== 6c. COMPANY START-DATE OUTLIERS (in-universe founding check: Sarvam AI / Krutrim) ===")
# uses job_companies + raw start years would need JSONL; approximate via career_start of
# candidates whose FIRST job is at the company -> instead parse JSONL once below if dumping.
print("(see JSONL pass below when --dump-records)")

print("\n=== 6d. DUPLICATE JOB-DESCRIPTION-WITHIN-PROFILE PREVALENCE ===")
def dup_share(jd):
    parts = [p.strip() for p in jd.split(" ||| ") if p.strip()]
    if len(parts) < 2:
        return False
    return len(set(parts)) < len(parts)
# job_descriptions joined with unknown sep; detect repeated 80-char chunk instead
first80 = df["job_descriptions"].str[:80]
def has_internal_dup(s):
    if not s or len(s) < 200:
        return False
    probe = s[:120]
    return s.count(probe) >= 2
df["desc_dup"] = df["job_descriptions"].apply(has_internal_dup)
print(f"Profiles where first 120 chars of job_descriptions repeat later (within-profile template dup): "
      f"{int(df['desc_dup'].sum())} ({df['desc_dup'].mean():.1%})")
print(f"  among T_ELITE: {df.loc[is_elite_t,'desc_dup'].mean():.1%} | among T_SOLID: {df.loc[is_solid_t,'desc_dup'].mean():.1%}")

# --------------------------- 7. THE 21 T_ELITE PROFILES + TEMPLATE FAMILY MAP
print("\n=== 7. ALL T_ELITE PROFILES (likely tier-5 core / honeypot mimics) ===")
te = df[is_elite_t].copy()
te["sal_inverted"] = te["sig_salary_min"] > te["sig_salary_max"]
cols7 = ["candidate_id", "current_title", "current_company", "yoe", "location", "country",
         "days_since_active", "sig_response_rate", "sig_notice_period_days",
         "sig_salary_min", "sig_salary_max", "sig_open_to_work", "sig_willing_to_relocate",
         "date_duration_mismatch", "n_expert_zero_months", "signup_after_active", "sal_inverted"]
print(te[cols7].sort_values("days_since_active").to_string(index=False))

print("\n=== 7b. GLOBAL SUMMARY-TEMPLATE FAMILY MAP (first 48 chars after name strip) ===")
fam = df["summary"].str.replace(r"^[A-Za-z /]+ with \d+(\.\d+)? ?\+? years? of ", "", regex=True) \
                   .str[:48]
vc = fam.value_counts().head(20)
print(vc.to_string())

print("\n=== 7c. T_SOLID (n=150) signal spread (candidate tier-4/plain-tier-5 band?) ===")
ts = df[is_solid_t]
print(ts.groupby(pd.cut(ts["days_since_active"], [0, 30, 60, 90, 180, 400]), observed=True)
        .size().to_string())
good_ts = ts[(ts["days_since_active"] <= 90) & (ts["sig_response_rate"] >= 0.5)
             & (ts["sig_notice_period_days"] <= 60)
             & (ts["country"].str.lower().eq("india") | ts["sig_willing_to_relocate"])]
print(f"T_SOLID fully-available India/relocate subset: {len(good_ts)}")
print("\nT_SOLID + T_ELITE available subset total:",
      len(good_ts) + int(((te['days_since_active']<=90) & (te['sig_response_rate']>=0.5)
                          & (te['sig_notice_period_days']<=60)
                          & (te['country'].str.lower().eq('india') | te['sig_willing_to_relocate'])).sum()))

print("\n=== 7d. PLAIN-LANGUAGE TIER-5 FAMILY (n=8, de-buzzworded retrieval people) ===")
T_PLAIN = "Senior engineer who has spent the last several y"
pl = df[df["summary"].str.contains(T_PLAIN, regex=False)]
print(pl[["candidate_id", "headline", "current_company", "yoe", "location",
          "days_since_active", "sig_response_rate", "sig_notice_period_days",
          "sig_salary_min", "sig_salary_max", "sig_assessment_mean",
          "date_duration_mismatch", "n_expert_zero_months", "signup_after_active"]]
      .to_string(index=False))

print("\n=== 7e. APPLIED-ML JUNIOR FAMILY (n~1000) — likely tier-2/3 band ===")
am = df[df["summary"].str.contains("experience in applied machine learning. Worked a", regex=False)]
print(f"n={len(am)} | india={am['country'].str.lower().eq('india').mean():.0%} "
      f"| med yoe={am['yoe'].median():.1f} | med active={am['days_since_active'].median():.0f}d "
      f"| med resp={am['sig_response_rate'].median():.2f} "
      f"| med sal_max={am['sig_salary_max'].median():.1f}")

if "--company-dates" in sys.argv:
    # One pass over raw JSONL: collect start years for stints at young AI companies
    # to test "tenure predates company founding" honeypot marker.
    targets = ["Sarvam AI", "Krutrim", "Rephrase.ai", "Yellow.ai", "Observe.AI", "Wysa"]
    pat = re.compile(r'\{"company": "(' + "|".join(re.escape(t) for t in targets)
                     + r')", "title": "[^"]*", "start_date": "(\d{4})-')
    from collections import Counter, defaultdict
    years = defaultdict(Counter)
    early_ids = defaultdict(list)
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            for m in pat.finditer(line):
                comp, yr = m.group(1), int(m.group(2))
                years[comp][yr] += 1
                if yr <= 2020 and comp in ("Sarvam AI", "Krutrim") and len(early_ids[comp]) < 15:
                    early_ids[comp].append(line[18:30])
    for comp in targets:
        c = years[comp]
        tot = sum(c.values())
        if tot == 0:
            continue
        lo = min(c); hi = max(c)
        pre2021 = sum(v for y, v in c.items() if y <= 2020)
        print(f"{comp:<12} stints={tot:>5} start-year range {lo}-{hi} | <=2020 starts: {pre2021} "
              f"({pre2021/tot:.0%})")
        print("   year hist:", dict(sorted(c.items())))
    print("Early (<=2020) Sarvam/Krutrim candidate ids:", dict(early_ids))

if "--dump-records" in sys.argv:
    want = set(DEEP_READ_IDS) | set(BAIT_IDS)
    found = {}
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            # cheap pre-check before json parse
            cid = line[18:30]
            if cid in want or any(w in line[:40] for w in want):
                rec = json.loads(line)
                if rec["candidate_id"] in want:
                    found[rec["candidate_id"]] = rec
                    if len(found) == len(want):
                        break
    out = "c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons/redrob-ranker/eda/exemplar_records.json"
    order = DEEP_READ_IDS + BAIT_IDS
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"deep_read": [found[i] for i in order if i in found and i in DEEP_READ_IDS],
                   "bait": [found[i] for i in order if i in found and i in BAIT_IDS]},
                  f, indent=1)
    print(f"Dumped {len(found)}/{len(want)} records to {out}")
