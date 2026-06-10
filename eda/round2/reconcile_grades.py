# -*- coding: utf-8 -*-
"""Grade reconciliation: 3-judge median vs round-1 PARA_GRADES; parquet impact; consensus JSON."""
import json
import re
import statistics
import pandas as pd

BASE = "c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons/redrob-ranker"
PARQUET = BASE + "/data/candidates_flat.parquet"
PARAS_44 = BASE + "/eda/round2/paragraphs_44.json"
TIER5 = BASE + "/eda/tier5-language.py"
OUT = BASE + "/eda/round2/paragraph_grades_consensus.json"

J1 = {"P00":0,"P01":0,"P02":0,"P03":0,"P04":0,"P05":0,"P06":0,"P07":0,"P08":0,"P09":1,
      "P10":0,"P11":0,"P12":1,"P13":0,"P14":0,"P15":1,"P16":2,"P17":2,"P18":2,"P19":2,
      "P20":2,"P21":3,"P22":4,"P23":2,"P24":3,"P25":3,"P26":3,"P27":5,"P28":5,"P29":5,
      "P30":3,"P31":5,"P32":4,"P33":5,"P34":5,"P35":5,"P36":5,"P37":5,"P38":5,"P39":5,
      "P40":5,"P41":5,"P42":5,"P43":5}
J2 = {"P00":0,"P01":0,"P02":0,"P03":0,"P04":0,"P05":0,"P06":0,"P07":0,"P08":0,"P09":1,
      "P10":1,"P11":1,"P12":1,"P13":1,"P14":1,"P15":2,"P16":2,"P17":2,"P18":2,"P19":2,
      "P20":2,"P21":3,"P22":3,"P23":2,"P24":3,"P25":3,"P26":3,"P27":4,"P28":4,"P29":4,
      "P30":3,"P31":4,"P32":3,"P33":5,"P34":5,"P35":5,"P36":5,"P37":5,"P38":5,"P39":5,
      "P40":5,"P41":5,"P42":5,"P43":5}
J3 = {"P00":0,"P01":0,"P02":0,"P03":0,"P04":0,"P05":0,"P06":0,"P07":0,"P08":0,"P09":1,
      "P10":1,"P11":1,"P12":1,"P13":1,"P14":1,"P15":2,"P16":2,"P17":2,"P18":2,"P19":2,
      "P20":2,"P21":3,"P22":3,"P23":2,"P24":3,"P25":3,"P26":3,"P27":4,"P28":4,"P29":4,
      "P30":3,"P31":4,"P32":3,"P33":5,"P34":5,"P35":5,"P36":5,"P37":5,"P38":5,"P39":5,
      "P40":5,"P41":5,"P42":5,"P43":5}

# Round-1 flagged uncertain (EDA_SYNTHESIS section 6, item 4)
FLAGGED7 = {"P30", "P22", "P23", "P21", "P25", "P26", "P32"}

# Final-call notes for flagged/contested paragraphs (one line each)
NOTES = {
    "P30": "RAG support chatbot = JD named anti-pattern; real engineering (Pinecone pipeline, cost fine-tune, 31% impact) but eval is BLEU/ROUGE generation metrics not retrieval/ranking quality -> 3 (round-1's own flag said 2-3; demoted from 4).",
    "P22": "Production recsys modeling (CF + GBM re-rank) is JD-core domain, but 'lighter weight' hedge, no eval evidence, deployment owned by platform team -> 3 (demoted from round-1's 4).",
    "P23": "CV-only fine-tuning with self-declared 'limited' NLP experience is the JD's explicit disqualifier pattern; real PyTorch production work keeps it at 2, unanimous.",
    "P21": "Production-side ML engineering (serving API, feature store, observability) for fraud detection; modeling secondary, zero IR -> 3 confirmed, unanimous.",
    "P25": "Churn/LTV XGBoost actually used by retention team but 'more modeling than productionization', no IR -> 3 confirmed, unanimous.",
    "P26": "Transformer-based NLP classification (DistilBERT, HF) gives genuine NLP exposure but internal dashboard, no retrieval/ranking -> 3 confirmed, unanimous.",
    "P32": "Strong MLOps ownership (drift detection, monitoring, Kubeflow) is operationally JD-adjacent but only application is churn, zero IR/ranking -> 3 (median over one judge's 4).",
}
DIFF_NOTES = {  # non-flagged places where consensus moved off round-1 by 1
    "P27": "5->4: dead-center LTR search ranking + relevance labeling + eval, but no embeddings/vector-search must-have evidence (2/3 judges).",
    "P28": "5->4: ranking models + offline-online correlation (must-have #4) but no embeddings/retrieval stack (2/3 judges).",
    "P29": "5->4: sentence-transformers + FAISS + human-judged eval hits must-haves 1-2, but internal 500K KB, no production retrieval ops or ranking metrics (2/3 judges).",
    "P31": "5->4: 10M-user recsys with embeddings + A/B infra, lacks vector-DB ops and ranking-metric eval rigor (2/3 judges).",
    "P24": "2->3: shipped LightGBM forecasting model = narrower-but-real production ML, matches grade-3 anchor (unanimous 3).",
    "P09": "0->1: distributed infra/ops is a JD nice-to-have adjacency (unanimous 1).",
    "P10": "0->1: generic-engineer band kept at 1 for internal consistency (2/3 judges); immaterial to tier3+ cutoff.",
    "P11": "0->1: same engineer-no-ML band consistency call (2/3 judges).",
    "P12": "0->1: Kafka/distributed backend = JD nice-to-have adjacency (unanimous 1).",
    "P13": "0->1: engineer-no-ML band consistency (2/3 judges).",
    "P14": "0->1: engineer-no-ML band consistency (2/3 judges).",
    "P15": "1->2: strong data-infra one step below data-infra-with-ML; weakest 2 (2/3 judges).",
    "P16": "1->2: pipelines feeding internal ML models = data-infra-with-ML-integration anchor (unanimous 2).",
    "P17": "1->2: data infra + small shipped predictive features (unanimous 2).",
    "P18": "1->2: streaming feature pipelines aligned with DS models, 'adjacent ML exposure' (unanimous 2).",
}

paras = json.load(open(PARAS_44, encoding="utf-8"))
pid2text = {p["pid"]: p["text"] for p in paras}
pid2count = {p["pid"]: p["count"] for p in paras}

# ---- round-1 grades: parse PARA_GRADES from tier5-language.py, match by TEXT ----
src = open(TIER5, encoding="utf-8").read()
entry_re = re.compile(r'^\s*"(.+?)":\s*\("(P\d\d)",\s*(\d+),\s*"([^"]+)"\),?\s*$', re.M)
r1_entries = entry_re.findall(src)
assert len(r1_entries) == 44, f"parsed {len(r1_entries)} round-1 entries, expected 44"

r1_grade_by_pid = {}   # pid (paragraphs_44 numbering) -> round-1 grade
r1_pid_map = {}        # paragraphs_44 pid -> round-1 pid (to verify numbering)
for prefix, r1pid, g, label in r1_entries:
    matches = [pid for pid, t in pid2text.items() if t.startswith(prefix[:48])]
    assert len(matches) == 1, f"prefix {prefix[:48]!r} matched {matches}"
    pid = matches[0]
    r1_grade_by_pid[pid] = int(g)
    r1_pid_map[pid] = r1pid
assert len(r1_grade_by_pid) == 44
mismatch = [(k, v) for k, v in r1_pid_map.items() if k != v]
print(f"round-1 vs round-2 pid numbering mismatches: {mismatch if mismatch else 'NONE (identical numbering)'}")

# ---- consensus = median; contested per task rule + the 7 flagged ----
rows = []
contested_rule, flagged_rows, overrides = [], [], []
for p in paras:
    pid = p["pid"]
    g3 = [J1[pid], J2[pid], J3[pid]]
    med = int(statistics.median(g3))
    spread = max(g3) - min(g3)
    r1 = r1_grade_by_pid[pid]
    rule_contested = spread >= 2 or abs(med - r1) >= 2
    if rule_contested:
        contested_rule.append(pid)
    contested = rule_contested or pid in FLAGGED7
    final = med  # no median overrides; see report
    note = NOTES.get(pid) or DIFF_NOTES.get(pid) or ""
    if med != r1 and not note:
        note = f"consensus {med} vs round-1 {r1}"
    rows.append({"pid": pid, "text_prefix_60": p["text"][:60], "count": p["count"],
                 "grade": final, "contested": bool(contested), "note": note})
    if pid in FLAGGED7:
        flagged_rows.append((pid, g3, med, r1, final))

print(f"\nparagraphs contested by strict rule (spread>=2 or |median-round1|>=2): {contested_rule if contested_rule else 'NONE'}")
print("\n7 round-1-flagged paragraphs settled:")
print(f"{'pid':4} {'judges':>9} {'median':>6} {'round1':>6} {'final':>5}")
for pid, g3, med, r1, fin in flagged_rows:
    print(f"{pid:4} {str(g3):>9} {med:6} {r1:6} {fin:5}")

print("\nall consensus-vs-round-1 differences:")
for r in rows:
    r1 = r1_grade_by_pid[r["pid"]]
    if r["grade"] != r1:
        print(f"  {r['pid']}: round1={r1} -> consensus={r['grade']} (count={r['count']})")

json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"\nwrote {OUT} ({len(rows)} rows)")

# ---- parquet impact ----
df = pd.read_parquet(PARQUET, columns=["candidate_id", "job_descriptions"])
print(f"\nparquet rows: {len(df)}")
prefix2pid = {pid2text[pid][:48]: pid for pid in pid2text}
cons = {r["pid"]: r["grade"] for r in rows}

def best_grades(s):
    b1 = b2 = -1
    for para in s.split("\n"):
        para = para.strip()
        if not para:
            continue
        pid = prefix2pid.get(para[:48])
        assert pid is not None, f"unmatched paragraph: {para[:60]!r}"
        g1, g2 = r1_grade_by_pid[pid], cons[pid]
        if g1 > b1: b1 = g1
        if g2 > b2: b2 = g2
    return b1, b2

bg = df["job_descriptions"].fillna("").map(best_grades)
df["best_r1"] = bg.map(lambda t: t[0])
df["best_cons"] = bg.map(lambda t: t[1])
assert (df["best_r1"] >= 0).all()

print("\nbest-grade distribution ROUND-1:")
print(df["best_r1"].value_counts().sort_index().to_string())
print("\nbest-grade distribution CONSENSUS:")
print(df["best_cons"].value_counts().sort_index().to_string())

changed = (df["best_r1"] != df["best_cons"]).sum()
print(f"\ncandidates whose best grade changed: {changed}")
print("\ncrosstab round1 (rows) x consensus (cols):")
print(pd.crosstab(df["best_r1"], df["best_cons"]).to_string())

def strata(col):
    return {g: int((df[col] == g).sum()) for g in (5, 4, 3)} | {
        ">=4": int((df[col] >= 4).sum()), ">=3": int((df[col] >= 3).sum())}
print(f"\nROUND-1 strata: {strata('best_r1')}")
print(f"CONSENSUS strata: {strata('best_cons')}")

# movement across the {5 / 4 / 3 / <3} strata specifically
def stratum(g):
    return g if g >= 3 else 0
moved = (df["best_r1"].map(stratum) != df["best_cons"].map(stratum)).sum()
print(f"candidates changing {{5/4/3/<3}} stratum: {moved}")
