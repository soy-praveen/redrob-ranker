"""Synthesis cross-checks: resolve conflicts between specialist analysts.
Run: PYTHONIOENCODING=utf-8 python synthesis_check.py
"""
import pandas as pd

PQ = "c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons/redrob-ranker/data/candidates_flat.parquet"
HONEY = "c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons/redrob-ranker/eda/honeypot_candidates.csv"
STUFF = "c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons/redrob-ranker/eda/stuffer_ids.csv"

df = pd.read_parquet(PQ)
hp = pd.read_csv(HONEY)
hp_ids = set(hp[hp.columns[0]] if "candidate_id" not in hp.columns else hp["candidate_id"])
print(f"honeypot csv rows: {len(hp)}, ids: {len(hp_ids)}")

PLAIN8 = ["CAND_0005538","CAND_0006567","CAND_0030468","CAND_0037980",
          "CAND_0061257","CAND_0068351","CAND_0080766","CAND_0093193"]
ELITE_STRONG = ["CAND_0081846","CAND_0018499","CAND_0046525","CAND_0077337","CAND_0011687","CAND_0002025"]
GHOSTS = ["CAND_0007411","CAND_0033861","CAND_0041611","CAND_0060072","CAND_0092278","CAND_0094759"]

# 1) Do any tier-5 plain / elite-strong ids collide with the 93 honeypots?
print("plain8 in honeypots:", sorted(set(PLAIN8) & hp_ids))
print("elite_strong in honeypots:", sorted(set(ELITE_STRONG) & hp_ids))
print("ghosts in honeypots:", sorted(set(GHOSTS) & hp_ids))

# 2) Consulting-only discrepancy: 7,034 (census, six firms) vs 9,745 (exemplars)
SIX = ["TCS","Infosys","Wipro","Accenture","Cognizant","Capgemini"]
def consulting_only(jc):
    comps = [c.strip() for c in str(jc).split("|") if c.strip()]
    return len(comps) > 0 and all(any(s.lower() in c.lower() for s in SIX) for c in comps)
co6 = df["job_companies"].map(consulting_only)
print(f"consulting-ONLY (six JD firms): {co6.sum()}")
# broader: any company containing consulting-ish names? check distinct companies
allc = set()
for jc in df["job_companies"].dropna().head(20000):
    allc.update(c.strip() for c in str(jc).split("|"))
cons_like = sorted(c for c in allc if any(s.lower() in c.lower() for s in
                   ["tcs","infosys","wipro","accenture","cognizant","capgemini","hcl","tech mahindra","consult"]))
print("consulting-like company names in data:", cons_like)

# 3) Stuffer boilerplate count + overlap with honeypots
boiler = df["summary"].str.contains("online courses on RAG and vector databases", na=False)
print(f"stuffer boilerplate count: {boiler.sum()}")
stuff_ids = set(df.loc[boiler, "candidate_id"])
print(f"stuffer ∩ honeypots: {len(stuff_ids & hp_ids)}")

# 4) Honeypot overlap with the AI-titled pool (DQ-risk count)
AI_TITLES_RE = r"(?i)(ml engineer|machine learning|ai engineer|ai research|ai specialist|data scientist|nlp engineer|applied scientist|applied ml|search engineer|recommendation|staff ml|lead ai|\(ml\))"
ai_titled = df["current_title"].str.contains(AI_TITLES_RE, na=False)
print(f"AI-titled total: {ai_titled.sum()}")
hp_ai = df[df["candidate_id"].isin(hp_ids) & ai_titled]
print(f"honeypots with AI titles: {len(hp_ai)}")
print(hp_ai[["candidate_id","current_title","yoe","career_span_months"]].to_string(index=False))

# 5) Grade-5 head sizing: candidates whose job_descriptions contain elite paragraph markers
# T_ELITE summary template vs paragraph evidence (rough recheck of 21 vs 168 conflict)
t_elite = df["summary"].str.contains("production ML systems with a focus on search, retrieval, and ranking", na=False)
t_plain = df["summary"].str.contains("connect users with relevant information", na=False)
t_solid = df["summary"].str.contains("building ML-powered features in production", na=False)
print(f"T_ELITE summaries: {t_elite.sum()}, T_PLAIN: {t_plain.sum()}, T_SOLID: {t_solid.sum()}")
# paragraph-level grade-5-ish phrases in job_descriptions
g5 = df["job_descriptions"].str.contains(r"(?i)(NDCG|MRR|learning.to.rank|semantic search|BM25|recommendation (system|engine)|embedding|retrieval)", na=False)
print(f"strong-IR phrase in job_descriptions: {g5.sum()}")
print(f"  of which AI-titled: {(g5 & ai_titled).sum()}")
overlap = df[(t_elite | t_plain) & df["candidate_id"].isin(hp_ids)]
print(f"T_ELITE/T_PLAIN ∩ honeypots: {len(overlap)}")
if len(overlap): print(overlap[["candidate_id","current_title","yoe"]].to_string(index=False))
