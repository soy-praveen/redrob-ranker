# -*- coding: utf-8 -*-
"""
KEYWORD-STUFFER analysis for Redrob ranker EDA.

Questions:
 1. Size the stuffer population: non-technical current_title + >=5 AI skills.
 2. Which features separate stuffers from genuine AI-titled candidates?
    (AI-skill duration_months, endorsements, assessment scores, ML language
     in job_descriptions/summary)
 3. Print full raw-JSONL profiles for ~10 stuffers (archetype characterization).
 4. Cross-over check: non-AI titles whose job descriptions show REAL ML work.
 5. Verify sample_submission top-20 against stuffer definition.

Run:  PYTHONIOENCODING=utf-8 python stuffers.py [--profiles]
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
SAMPLE_SUB = ("c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons/hack2skill/"
              "The Data & AI Challenge/[PUB] India_runs_data_and_ai_challenge/"
              "India_runs_data_and_ai_challenge/sample_submission.csv")

# ---------------------------------------------------------------- keyword set
# Exact skill-name vocabulary is small (133 distinct). AI set built from it.
AI_SKILLS = {
    # retrieval / JD-core
    "Embeddings", "Vector Search", "Semantic Search", "Information Retrieval",
    "RAG", "FAISS", "Pinecone", "Weaviate", "Milvus", "Qdrant", "pgvector",
    "Elasticsearch", "OpenSearch", "BM25", "Learning to Rank",
    "Recommendation Systems", "Sentence Transformers", "Haystack", "LlamaIndex",
    # llm
    "LLMs", "Fine-tuning LLMs", "LoRA", "QLoRA", "PEFT", "Prompt Engineering",
    "LangChain", "Hugging Face Transformers",
    # classic ml
    "Machine Learning", "Deep Learning", "NLP", "Python", "PyTorch",
    "TensorFlow", "scikit-learn", "Data Science", "Feature Engineering",
    "Statistical Modeling", "Time Series", "Forecasting",
    "Reinforcement Learning",
    # mlops
    "MLOps", "MLflow", "Kubeflow", "BentoML", "Weights & Biases",
    # cv / speech (AI keywords even if off-JD)
    "Computer Vision", "OpenCV", "YOLO", "CNN", "GANs", "Diffusion Models",
    "Image Classification", "Object Detection", "Speech Recognition", "ASR",
    "TTS",
    # rare plain-language synonyms
    "Information Retrieval Systems", "Search Backend", "Text Encoders",
    "Vector Representations", "Content Matching", "Model Adaptation",
    "Ranking Systems", "Search & Discovery", "Search Infrastructure",
    "Indexing Algorithms", "Open-source ML libraries",
    "Natural Language Processing", "Document Processing",
}
# JD-core subset (retrieval/embeddings/ranking — what the JD actually needs)
JD_CORE = {
    "Embeddings", "Vector Search", "Semantic Search", "Information Retrieval",
    "RAG", "FAISS", "Pinecone", "Weaviate", "Milvus", "Qdrant", "pgvector",
    "Elasticsearch", "OpenSearch", "BM25", "Learning to Rank",
    "Recommendation Systems", "Sentence Transformers", "Haystack", "LlamaIndex",
    "LLMs", "Fine-tuning LLMs", "LoRA", "QLoRA", "PEFT",
    "Hugging Face Transformers", "NLP",
}

NONTECH_TITLES = {
    "Business Analyst", "HR Manager", "Mechanical Engineer", "Accountant",
    "Project Manager", "Customer Support", "Operations Manager",
    "Content Writer", "Sales Executive", "Civil Engineer", "Graphic Designer",
    "Marketing Manager",
}
TECH_NONAI_TITLES = {
    "Software Engineer", "Full Stack Developer", "Cloud Engineer",
    "Java Developer", ".NET Developer", "DevOps Engineer", "Mobile Developer",
    "Frontend Engineer", "QA Engineer",
}
DATA_TITLES = {
    "Analytics Engineer", "Data Engineer", "Data Analyst", "Backend Engineer",
    "Senior Data Engineer", "Senior Software Engineer", "Data Scientist",
    "Senior Data Scientist",
}
# everything else (ML Engineer, AI Research Engineer, NLP Engineer, ...) = AI

ML_TEXT_RE = re.compile(
    r"\b(machine.learning|deep.learning|neural net|NLP|natural.language|"
    r"embedding|vector (search|database|db|index|store)|retrieval|"
    r"semantic search|recommendation (system|engine|model|pipeline)|"
    r"recommender|learning.to.rank|re.?rank|fine.?tun\w*|"
    r"large.language.model|\bLLM\b|transformer|\bBERT\b|RAG\b|"
    r"retrieval.augmented|FAISS|pinecone|milvus|weaviate|qdrant|pgvector|"
    r"elasticsearch|opensearch|BM25|two.tower|\bANN\b|HNSW|"
    r"\bNDCG\b|\bMRR\b|model (training|serving|deployment)|MLOps|"
    r"sentence.transformer|hugging.?face|pytorch|tensorflow|scikit|"
    r"LoRA|PEFT|word2vec|tf.?idf)\b",
    re.IGNORECASE)


def title_cat(t):
    if t in NONTECH_TITLES:
        return "NONTECH"
    if t in TECH_NONAI_TITLES:
        return "TECH_NONAI"
    if t in DATA_TITLES:
        return "DATA"
    return "AI"


def main():
    df = pd.read_parquet(PARQUET)
    print(f"loaded {len(df)} rows")

    df["skills_list"] = df["skill_names"].fillna("").apply(
        lambda s: s.split(" | ") if s else [])
    df["ai_skill_n"] = df["skills_list"].apply(
        lambda L: sum(1 for k in L if k in AI_SKILLS))
    df["jd_core_n"] = df["skills_list"].apply(
        lambda L: sum(1 for k in L if k in JD_CORE))
    df["tcat"] = df["current_title"].apply(title_cat)

    # ------------------------------------------------ Q1: size the population
    print("\n=========== Q1: AI-skill count by title category ===========")
    print(df.groupby("tcat")["ai_skill_n"].describe().round(2).to_string())
    print("\nAI-skill count distribution per category (% of category):")
    tab = pd.crosstab(df["tcat"],
                      pd.cut(df["ai_skill_n"], [-1, 0, 2, 4, 7, 100],
                             labels=["0", "1-2", "3-4", "5-7", "8+"]),
                      normalize="index").round(3) * 100
    print(tab.to_string())

    print("\nPer-title counts of >=5 AI skills (non-technical titles):")
    sub = df[df["tcat"] == "NONTECH"]
    per = (sub.assign(stuf=sub["ai_skill_n"] >= 5)
           .groupby("current_title")["stuf"].agg(["sum", "count", "mean"]))
    per["mean"] = (per["mean"] * 100).round(1)
    print(per.to_string())

    n_stuff5 = int((df["tcat"].eq("NONTECH") & (df["ai_skill_n"] >= 5)).sum())
    n_stuff3 = int((df["tcat"].eq("NONTECH") & (df["ai_skill_n"] >= 3)).sum())
    n_stuff5_jd = int((df["tcat"].eq("NONTECH") & (df["jd_core_n"] >= 5)).sum())
    print(f"\nSTUFFER population (NONTECH title, >=5 AI skills): {n_stuff5}")
    print(f"  ... >=3 AI skills: {n_stuff3}")
    print(f"  ... >=5 JD-core (retrieval/LLM) skills:  {n_stuff5_jd}")
    n_tech5 = int((df["tcat"].eq("TECH_NONAI") & (df["ai_skill_n"] >= 5)).sum())
    print(f"TECH_NONAI with >=5 AI skills: {n_tech5}")

    print("\nExact NONTECH ai_skill_n value counts (bimodality check):")
    print(df.loc[df["tcat"] == "NONTECH", "ai_skill_n"]
          .value_counts().sort_index().to_string())

    # which AI skills do stuffers list? fraction JD-core?
    stuf_mask = df["tcat"].eq("NONTECH") & (df["ai_skill_n"] >= 5)
    from collections import Counter
    cnt = Counter()
    jd_frac = []
    for L in df.loc[stuf_mask, "skills_list"]:
        ai = [k for k in L if k in AI_SKILLS]
        cnt.update(ai)
        jd_frac.append(sum(1 for k in ai if k in JD_CORE) / len(ai))
    print(f"\nstuffers: mean fraction of their AI skills that are JD-core: "
          f"{np.mean(jd_frac):.3f}")
    print("top 25 AI skills among stuffers:")
    for k, v in cnt.most_common(25):
        print(f"  {v:5d}  {k}")

    # is there a tech-titled stuffer variant? jd_core_n distribution
    print("\njd_core_n >= 5 by category (stuffer-style JD-core stuffing):")
    for cat in ["NONTECH", "TECH_NONAI", "DATA", "AI"]:
        c = df[df["tcat"] == cat]
        print(f"  {cat:10s}: {int((c['jd_core_n'] >= 5).sum()):5d} of {len(c)}"
              f"  (jd_core median {c['jd_core_n'].median():.0f})")
    print("\nTECH_NONAI jd_core_n value counts:")
    print(df.loc[df["tcat"] == "TECH_NONAI", "jd_core_n"]
          .value_counts().sort_index().to_string())

    # boilerplate detectors discovered from raw profiles
    print("\nBoilerplate-phrase detectors (counts by category & stuffer overlap):")
    PHRASE = "online courses on RAG and vector databases"
    HEADLINE_PATS = ("Exploring AI & GenAI applications", "AI enthusiast",
                     "Generative AI explorer", "Building with LLMs")
    has_phrase = df["summary"].fillna("").str.contains(PHRASE, regex=False)
    has_head = df["headline"].fillna("").apply(
        lambda h: any(p in h for p in HEADLINE_PATS))
    for name, mask in [("summary phrase", has_phrase),
                       ("enthusiast headline", has_head)]:
        print(f"  {name}: total={int(mask.sum())}")
        print(df.loc[mask, "tcat"].value_counts().to_string())
        both = int((mask & stuf_mask).sum())
        print(f"    overlap with skill-stuffers: {both}/{int(stuf_mask.sum())} "
              f"stuffers, {int(mask.sum()) - both} non-stuffers also flagged")
    # do the two detectors agree with each other?
    print(f"  phrase & headline agree: "
          f"{int((has_phrase & has_head).sum())} both, "
          f"{int((has_phrase ^ has_head).sum())} xor")

    # are the few tech-titled jd_core>=5 candidates stuffer-like?
    print("\nTECH_NONAI / DATA candidates with jd_core_n >= 5 (stuffer-like?):")
    tt = df[df["tcat"].isin(["TECH_NONAI", "DATA"]) & (df["jd_core_n"] >= 5)]
    tt_phrase = int(has_phrase.loc[tt.index].sum())
    print(f"  n={len(tt)}, with boilerplate phrase: {tt_phrase}")

    # rare plain-language skill holders (potential hand-crafted tier-5s)
    RARE = {"Information Retrieval Systems", "Search Backend", "Text Encoders",
            "Vector Representations", "Content Matching", "Model Adaptation",
            "Ranking Systems", "Search & Discovery", "Search Infrastructure",
            "Indexing Algorithms", "Open-source ML libraries",
            "Natural Language Processing", "Document Processing",
            "Workflow Orchestration"}
    rare_mask = df["skills_list"].apply(lambda L: any(k in RARE for k in L))
    print(f"\nholders of rare plain-language skills: {int(rare_mask.sum())}")
    print(df.loc[rare_mask, ["candidate_id", "current_title", "yoe",
                             "skill_names"]].to_string(max_colwidth=80))

    df["is_stuffer"] = df["tcat"].eq("NONTECH") & (df["ai_skill_n"] >= 5)
    df["is_ai_titled"] = df["tcat"].eq("AI")

    # ------------------------------- Q2: separating features stuffer vs genuine
    print("\n=========== Q2: stuffers vs genuine AI-titled ===========")

    def ai_skill_stats(row):
        try:
            skills = json.loads(row["skills_json"])
        except (TypeError, ValueError):
            return pd.Series([np.nan] * 6)
        ai = [s for s in skills if s.get("name") in AI_SKILLS]
        if not ai:
            return pd.Series([np.nan] * 6)
        dur = [s.get("duration_months") or 0 for s in ai]
        end = [s.get("endorsements") or 0 for s in ai]
        prof_hi = sum(1 for s in ai
                      if s.get("proficiency") in ("advanced", "expert"))
        zero = sum(1 for d in dur if d == 0)
        return pd.Series([np.mean(dur), np.max(dur), np.mean(end),
                          prof_hi / len(ai), zero, len(ai)])

    work = df[df["is_stuffer"] | df["is_ai_titled"]].copy()
    cols = ["ai_dur_mean", "ai_dur_max", "ai_end_mean", "ai_hiprof_frac",
            "ai_zero_dur_n", "ai_n"]
    work[cols] = work.apply(ai_skill_stats, axis=1)

    # assessments: scores restricted to AI-skill assessments
    def ai_assess(row):
        try:
            a = json.loads(row["sig_assessments_json"])
        except (TypeError, ValueError):
            return pd.Series([0, np.nan])
        ai = {k: v for k, v in a.items() if k in AI_SKILLS}
        if not ai:
            return pd.Series([0, np.nan])
        return pd.Series([len(ai), np.mean(list(ai.values()))])

    work[["ai_assess_n", "ai_assess_mean"]] = work.apply(ai_assess, axis=1)

    # for the 12% of stuffers WITH assessments: score distribution
    sa = work[work["is_stuffer"] & (work["ai_assess_n"] > 0)]
    ga = work[work["is_ai_titled"] & (work["ai_assess_n"] > 0)]
    print(f"\nstuffers WITH >=1 AI assessment: n={len(sa)}, "
          f"score quantiles {sa['ai_assess_mean'].quantile([.1,.5,.9]).round(1).tolist()}")
    print(f"genuine WITH >=1 AI assessment: n={len(ga)}, "
          f"score quantiles {ga['ai_assess_mean'].quantile([.1,.5,.9]).round(1).tolist()}")

    # ML language in free text
    def ml_text(row):
        txt = " ".join(str(row[c]) for c in
                       ("summary", "headline", "job_descriptions"))
        return len(ML_TEXT_RE.findall(txt))

    work["ml_text_hits"] = work.apply(ml_text, axis=1)
    work["ml_in_text"] = work["ml_text_hits"] > 0

    # term-level diagnosis: which regex terms fire for stuffers vs genuine?
    print("\nWhich ML terms fire (pct of group with >=1 match of that term):")
    term_res = {
        "embedding": r"embedding", "retrieval": r"retriev",
        "vector_db": r"vector (search|database|db|index|store)|faiss|pinecone|milvus|weaviate|qdrant|pgvector",
        "semantic_search": r"semantic search", "recsys": r"recommendation (system|engine|model)|recommender",
        "ranking_eval": r"\bNDCG\b|\bMRR\b|learning.to.rank|re.?rank",
        "llm_finetune": r"fine.?tun|LoRA|PEFT|large.language.model|\bLLM",
        "ml_generic": r"machine.learning|deep.learning|pytorch|tensorflow|scikit",
        "nlp": r"\bNLP\b|natural.language|transformer|\bBERT\b",
        "model_deploy": r"model (training|serving|deployment)|MLOps",
    }
    for name, pat in term_res.items():
        rx = re.compile(pat, re.IGNORECASE)
        fire = work.apply(lambda r: bool(rx.search(
            " ".join(str(r[c]) for c in ("summary", "headline",
                                         "job_descriptions")))), axis=1)
        gtab = fire.groupby(work["is_stuffer"]).mean() * 100
        print(f"  {name:16s} genuine {gtab.get(False, 0):5.1f}%  "
              f"stuffer {gtab.get(True, 0):5.1f}%")

    grp = work.groupby("is_stuffer")
    feats = cols + ["ai_assess_n", "ai_assess_mean", "ml_text_hits",
                    "total_skill_endorsements", "n_expert_zero_months", "yoe"]
    print("\nmeans (False = genuine AI-titled, True = stuffer):")
    print(grp[feats].mean().round(2).T.to_string())
    print("\nmedians:")
    print(grp[feats].median().round(2).T.to_string())
    print("\npct with ANY ML language in summary/headline/job_descriptions:")
    print((grp["ml_in_text"].mean() * 100).round(1).to_string())
    pct_assessed = grp.apply(
        lambda g: (g["ai_assess_n"] > 0).mean() * 100, include_groups=False)
    print("\npct with >=1 AI assessment:")
    print(pct_assessed.round(1).to_string())

    # quantiles to find clean thresholds
    print("\nQuantiles of key separators:")
    for f in ["ai_dur_mean", "ai_end_mean", "ai_assess_mean", "ml_text_hits"]:
        q = grp[f].quantile([.1, .25, .5, .75, .9]).unstack().round(1)
        print(f"\n  {f}:")
        print(q.to_string())

    # JD-core-only duration/endorsement boundaries (tightest separators)
    def jd_dur_end(sj):
        try:
            sk = json.loads(sj)
        except (TypeError, ValueError):
            return pd.Series([np.nan, np.nan])
        ai = [x for x in sk if x.get("name") in JD_CORE]
        if not ai:
            return pd.Series([np.nan, np.nan])
        return pd.Series([np.mean([x["duration_months"] for x in ai]),
                          np.mean([x["endorsements"] for x in ai])])

    work[["jd_dur", "jd_end"]] = work["skills_json"].apply(jd_dur_end)
    sw = work[work["is_stuffer"]]
    gw = work[~work["is_stuffer"]]
    print("\nJD-core skill mean-duration boundary:")
    print(f"  stuffer max {sw['jd_dur'].max():.1f}  p99 "
          f"{sw['jd_dur'].quantile(.99):.1f}  median {sw['jd_dur'].median():.1f}")
    print(f"  genuine min {gw['jd_dur'].min():.1f}  p1 "
          f"{gw['jd_dur'].quantile(.01):.1f}  median {gw['jd_dur'].median():.1f}")
    print("JD-core skill mean-endorsement boundary:")
    print(f"  stuffer max {sw['jd_end'].max():.1f}  p99 "
          f"{sw['jd_end'].quantile(.99):.1f}  median {sw['jd_end'].median():.1f}")
    print(f"  genuine min {gw['jd_end'].min():.1f}  p1 "
          f"{gw['jd_end'].quantile(.01):.1f}  median {gw['jd_end'].median():.1f}")
    for t in [16, 18, 20, 24]:
        print(f"  jd_dur < {t}: stuffers {(sw['jd_dur'] < t).mean()*100:.2f}%"
              f"  genuine {(gw['jd_dur'] < t).mean()*100:.2f}%")

    # behavioral-signal envelope probe
    print("\nbehavioral signal medians (stuffer vs genuine AI-titled):")
    for c in ["sig_response_rate", "sig_profile_completeness",
              "sig_github_activity", "yoe"]:
        print(f"  {c:26s} {sw[c].median():8.2f}  {gw[c].median():8.2f}")

    # single-threshold separation power
    print("\nThreshold separation checks:")
    s, g = work[work["is_stuffer"]], work[~work["is_stuffer"]]
    for desc, fn in [
        ("ml_text_hits == 0", lambda d: d["ml_text_hits"].eq(0)),
        ("ai_assess_mean < 50", lambda d: d["ai_assess_mean"] < 50),
        ("ai_dur_mean < 24", lambda d: d["ai_dur_mean"] < 24),
        ("ai_end_mean < 15", lambda d: d["ai_end_mean"] < 15),
        ("no AI assessment", lambda d: d["ai_assess_n"].eq(0)),
    ]:
        print(f"  {desc:24s} stuffers {fn(s).mean()*100:5.1f}%  "
              f"genuine {fn(g).mean()*100:5.1f}%")

    # ----------------------------------------- Q4: genuine career-changers?
    print("\n=========== Q4: non-AI titles with REAL ML work in job descriptions ===========")
    # strict regex: evidence of *doing* retrieval/ML work, applied to
    # job_descriptions ONLY (summaries carry GenAI-excitement boilerplate)
    BUILT_ML_RE = re.compile(
        r"(embedding|vector (search|database|db|index|store)|semantic search|"
        r"retrieval|recommendation (system|engine|model|pipeline)|recommender|"
        r"learning.to.rank|re.?rank|ranking model|search relevance|"
        r"\bNDCG\b|\bMRR\b|fine.?tun\w+|\bRAG\b|retrieval.augmented|"
        r"FAISS|pinecone|milvus|weaviate|qdrant|pgvector|BM25|two.tower|"
        r"sentence.transformer|hugging.?face|\bBERT\b|word2vec|"
        r"(trained|built|deployed|shipped|productioni[sz]ed).{0,60}"
        r"(model|classifier|ranker))",
        re.IGNORECASE)

    def built_hits(r):
        return len(BUILT_ML_RE.findall(str(r["job_descriptions"])))

    df["built_ml_hits"] = df.apply(built_hits, axis=1)
    for cat in ["NONTECH", "TECH_NONAI", "DATA", "AI"]:
        c = df[df["tcat"] == cat]
        for thr in [1, 2, 4]:
            n = int((c["built_ml_hits"] >= thr).sum())
            print(f"  {cat:10s} built_ml_hits >= {thr}: {n:6d} "
                  f"({n/len(c)*100:5.2f}% of {len(c)})")

    # crossover population: non-AI title + real ML in jobdesc
    cross = df[df["tcat"].isin(["NONTECH", "TECH_NONAI"]) &
               (df["built_ml_hits"] >= 2)]
    n_cross_stuf = int(cross["is_stuffer"].sum())
    print(f"\n  non-AI-titled with >=2 built-ML hits: {len(cross)} "
          f"(of which {n_cross_stuf} are also skill-stuffers)")
    print("\n  examples (career-changer check, up to 6):")
    for _, r in cross.nlargest(6, "built_ml_hits").iterrows():
        print(f"\n  -- {r['candidate_id']} | {r['current_title']} | "
              f"yoe={r['yoe']} | hits={r['built_ml_hits']} | "
              f"ai_skills={r['ai_skill_n']} | stuffer={r['is_stuffer']}")
        print("     jobdesc:",
              str(r["job_descriptions"])[:400].replace("\n", " "))

    # do STUFFERS have any built-ML evidence in jobdescs?
    print(f"\n  stuffers with built_ml_hits == 0: "
          f"{int((df.loc[df['is_stuffer'], 'built_ml_hits'] == 0).sum())}"
          f" / {int(df['is_stuffer'].sum())}")
    print(f"  genuine AI-titled with built_ml_hits == 0: "
          f"{int((df.loc[df['is_ai_titled'], 'built_ml_hits'] == 0).sum())}"
          f" / {int(df['is_ai_titled'].sum())}")

    # -------------------------------------------- Q5: sample_submission top-20
    print("\n=========== Q5: sample_submission top-20 ===========")
    sub20 = pd.read_csv(SAMPLE_SUB).head(20)
    m = sub20.merge(df, on="candidate_id", how="left")
    m["ml_text_hits"] = m.apply(ml_text, axis=1)
    n_stuf = 0
    for _, r in m.iterrows():
        tag = "STUFFER" if (r["tcat"] == "NONTECH" and r["ai_skill_n"] >= 5) \
            else r["tcat"]
        n_stuf += tag == "STUFFER"
        print(f"  #{int(r['rank']):2d} {r['candidate_id']} "
              f"{r['current_title']:<22s} ai_skills={int(r['ai_skill_n']):2d} "
              f"jd_core={int(r['jd_core_n']):2d} ml_text={int(r['ml_text_hits']):2d} "
              f"yoe={r['yoe']:.1f}  -> {tag}")
    print(f"\n  top-20 stuffers by our definition: {n_stuf}/20")
    full_sub = pd.read_csv(SAMPLE_SUB)
    print(f"  sample_submission rows: {len(full_sub)}")
    mfull = full_sub.merge(df[["candidate_id", "tcat", "ai_skill_n",
                               "current_title", "yoe"]], on="candidate_id")
    n100 = int(((mfull["tcat"] == "NONTECH") &
                (mfull["ai_skill_n"] >= 5)).head(100).sum())
    print(f"  full sample_submission top-100 stuffers: {n100}/"
          f"{min(100, len(mfull))}")
    print("  tcat distribution across whole sample_submission:")
    print(mfull["tcat"].value_counts().to_string())

    # does the reasoning column match the actual candidate?
    print("\n  reasoning-vs-data consistency check (top 10):")
    n_title_match = 0
    for _, r in mfull.head(10).iterrows():
        claimed_title = str(r["reasoning"]).split(" with ")[0]
        match = claimed_title.lower() in str(r["current_title"]).lower()
        n_title_match += match
        print(f"   {r['candidate_id']}: claimed '{claimed_title}' "
              f"actual '{r['current_title']}' match={match} | "
              f"claimed_yrs={str(r['reasoning']).split(' with ')[1][:8]} "
              f"actual_yoe={r['yoe']}")
    print(f"   title matches: {n_title_match}/10")
    # id range of submission picks
    ids = mfull["candidate_id"].str.extract(r"(\d+)").astype(int)[0]
    print(f"   submission id range: {ids.min()}..{ids.max()}")

    # ------------------------------------------------- Q3: raw JSONL profiles
    if "--profiles" in sys.argv:
        print("\n=========== Q3: full raw profiles of 10 stuffers ===========")
        picks = (df[df["is_stuffer"]]
                 .sort_values("ai_skill_n", ascending=False)
                 .groupby("current_title").head(1)
                 .head(10)["candidate_id"].tolist())
        picked = set(picks)
        found = {}
        with open(JSONL, encoding="utf-8") as f:
            for line in f:
                # cheap pre-filter before json parse
                if not any(p in line for p in picked):
                    continue
                rec = json.loads(line)
                cid = rec.get("candidate_id")
                if cid in picked:
                    found[cid] = rec
                    picked.discard(cid)
                    if not picked:
                        break
        for cid in picks:
            rec = found.get(cid)
            if rec is None:
                continue
            print("\n" + "=" * 70)
            print(json.dumps(rec, indent=1)[:4500])

    # save stuffer ids for the ranker
    out = "c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons/redrob-ranker/eda/stuffer_ids.csv"
    df.loc[df["is_stuffer"], ["candidate_id"]].to_csv(out, index=False)
    print(f"\nsaved {int(df['is_stuffer'].sum())} stuffer ids -> {out}")


if __name__ == "__main__":
    main()
