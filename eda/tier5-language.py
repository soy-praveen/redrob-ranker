# -*- coding: utf-8 -*-
"""
Plain-language tier-5 analysis for the Redrob ranker.

KEY STRUCTURAL DISCOVERY: all 300,171 job-description paragraphs in the 100k corpus
come from a pool of just 44 distinct paragraphs, and all summaries from 76 normalized
templates. So 'plain-language tier-5 detection' is exact: fingerprint which paragraphs
a candidate holds and grade the 44 paragraphs once. The phrase lists below are still
produced (deliverable for a generalizing scorer) and are validated against the pool.

Run:  PYTHONIOENCODING=utf-8 python eda/tier5-language.py
"""
import re
import json
import collections
import pandas as pd

PARQUET = "c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons/redrob-ranker/data/candidates_flat.parquet"
JSONL = ("c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons/hack2skill/The Data & AI Challenge/"
         "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl")

# ----------------------------------------------------------------------------------
# DELIVERABLE 6a: hand-graded relevance of the 44-paragraph pool (key = first 60 chars)
# grade: 5 = core ranking/retrieval/search/recsys production work (JD bullseye)
#        4 = production embeddings/recsys adjacent (strong)
#        3 = applied ML in production, not IR (supporting)
#        2 = ML-lite / CV-only / forecasting (weak; CV-only is a JD disqualifier alone)
#        1 = data engineering adjacency
#        0 = irrelevant
# ----------------------------------------------------------------------------------
PARA_GRADES = {  # first 60 chars -> (id, grade, label)
    "Enterprise sales of cloud software solutions into the mid-m": ("P00", 0, "sales"),
    "Customer support team lead at a SaaS product. Managed a tea": ("P01", 0, "support"),
    "Marketing leadership role at a B2B SaaS company. Owned the ": ("P02", 0, "marketing"),
    "Business analyst at a consulting firm, working primarily wi": ("P03", 0, "biz-analyst"),
    "Brand design and creative direction at a consumer-products ": ("P04", 0, "design"),
    "Mechanical engineering design role at a hardware-product co": ("P05", 0, "mech"),
    "Senior accounting role at a mid-sized company — month-end c": ("P06", 0, "accounting"),
    "Content writing and SEO strategy for a tech-focused publica": ("P07", 0, "content-seo"),
    "Operations management role at a logistics company. Owned da": ("P08", 0, "ops"),
    "Cloud infrastructure and DevOps work at an enterprise SaaS ": ("P09", 0, "devops"),
    "Android mobile development using Java and (more recently) K": ("P10", 0, "android"),
    "Frontend engineering at a media company. React, TypeScript,": ("P11", 0, "frontend"),
    "Java backend development at a large enterprise — Spring Boo": ("P12", 0, "java-backend"),
    "Full-stack web application development at a SaaS company. B": ("P13", 0, "fullstack"),
    "Test automation and QA engineering for a fintech product. B": ("P14", 0, "qa"),
    "Designed and maintained the analytical data warehouse on Sn": ("P15", 1, "dw-sql"),
    "Built and maintained data pipelines on Apache Airflow proce": ("P16", 1, "airflow-de"),
    "Backend + data hybrid role at a growth-stage startup. Built": ("P17", 1, "backend-data"),
    "Implemented streaming data pipelines on Kafka and Spark Str": ("P18", 1, "streaming-de"),
    "Mixed data science and analytics-engineering role at a mark": ("P19", 2, "ds-mix-abtest"),
    "Backend development with Python (FastAPI), PostgreSQL, and ": ("P20", 2, "be-model-integr"),
    "Contributed to ML feature engineering and model deployment ": ("P21", 3, "ml-prod-side"),
    "Built recommendation-style features at a mid-stage startup ": ("P22", 4, "recsys-light"),
    "Built computer vision models for our product's image modera": ("P23", 2, "cv-only"),
    "Worked on time-series forecasting models for supply-chain d": ("P24", 2, "forecasting"),
    "Worked on customer-facing predictive modeling for an e-comm": ("P25", 3, "churn-ml"),
    "Built NLP pipelines for sentiment analysis and document cla": ("P26", 3, "nlp-classif"),
    "Owned the ranking layer for an e-commerce search product, e": ("P27", 5, "ltr-search"),
    "Trained and shipped multiple ranking models for our product": ("P28", 5, "feed-ranking"),
    "Developed a semantic search feature for an internal knowled": ("P29", 5, "semantic-search"),
    "Implemented a RAG-based customer support chatbot integrated": ("P30", 4, "rag-chatbot"),
    "Built a content recommendation system serving 10M+ users th": ("P31", 5, "recsys-10M"),
    "Built and operated production ML pipelines using MLflow for": ("P32", 3, "mlops-churn"),
    "Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA an": ("P33", 5, "lora-jd-match"),
    "Built a RAG-based ranking pipeline serving 50M+ queries per": ("P34", 5, "rag-rank-recruiter"),
    "Built and shipped a production recommendation system at a m": ("P35", 5, "recsys-marketplace"),
    "Owned the end-to-end ranking pipeline at a recommendations-": ("P36", 5, "rank-pipeline-e2e"),
    "Owned the design and rollout of a large-scale semantic sear": ("P37", 5, "semantic-35M"),
    "Led the migration from keyword-based to embedding-based sea": ("P38", 5, "embed-search-recruiter"),
    "Built systems that understand what users are looking for an": ("P39", 5, "PLAIN-matching"),
    "Shipped the personalization infrastructure: the system that": ("P40", 5, "PLAIN-personalization"),
    "Designed the ranking layer for the company's flagship produ": ("P41", 5, "PLAIN-ranking-layer"),
    "Owned the search and discovery experience end-to-end at a c": ("P42", 5, "PLAIN-search-disc"),
    "Led the engineering team building infrastructure to surface": ("P43", 5, "PLAIN-relevance-infra"),
}
PLAIN_IDS = {"P39", "P40", "P41", "P42", "P43"}

# ----------------------------------------------------------------------------------
# DELIVERABLE 6b: phrase vocabulary for a generalizing scorer.
# Matched case-insensitively against job_descriptions (+summary where noted).
# ----------------------------------------------------------------------------------
STRONG_PHRASES = [  # tier A — core ranking/retrieval/recsys/search evidence
    r"recommendation (?:system|engine|model|pipeline|service|algorithm)s?",
    r"recommendation-style features", r"recommender(?: system)?s?",
    r"personali[sz]ed recommendations?", r"collaborative filtering",
    r"content[- ]based (?:filtering|ranking)", r"matrix factori[sz]ation",
    r"implicit[- ]feedback", r"cold[- ]start", r"exploration[- ]exploitation",
    r"search ranking", r"ranking (?:model|system|pipeline|algorithm|layer|service|function|calibration)s?",
    r"learning[- ]to[- ]rank", r"\bltr\b", r"re[- ]?rank(?:er|ing)\b", r"feed ranking",
    r"ranking metrics", r"relevance labeling", r"discovery feed",
    r"information retrieval", r"retrieval (?:system|pipeline|quality|latency|layer|stage|service)s?",
    r"candidate retrieval", r"document retrieval", r"retrieval[- ]augmented", r"\brag[- ]based\b",
    r"semantic search", r"vector search", r"vector (?:database|db|index|store)s?",
    r"dense (?:retrieval|vectors?)", r"sparse (?:and dense|vectors?)", r"hybrid (?:search|setup|retrieval)",
    r"similarity search", r"(?:approximate )?nearest[- ]neigh?bou?rs?", r"\bhnsw\b", r"\bfaiss\b",
    r"\bbm25\b", r"\bpinecone\b", r"\bweaviate\b", r"\bqdrant\b", r"\bpgvector\b", r"\bmilvus\b",
    r"query (?:understanding|expansion|rewriting|parsing|reformulation)", r"vocabulary mismatch",
    r"search relevance", r"relevance (?:model|tuning|scoring|signals?|feedback|engineering|judgments?)",
    r"search (?:quality|infrastructure|platform|backend|stack)", r"search and discovery",
    r"embedding(?:s)?\b", r"two[- ]tower", r"bi[- ]encoder", r"cross[- ]encoder", r"dual encoder",
    r"sentence[- ]transformers?", r"\bsbert\b", r"\bbge\b", r"minilm", r"mpnet",
    r"matching (?:engine|algorithm|system|model|pipeline|logic|layer|score)s?",
    r"candidate[- ]jd matching", r"(?:most )?relevant (?:matches|results|content)",
    r"\bndcg\b", r"\bmrr\b", r"recall@\w+", r"precision@\w+", r"\bdcg\b",
    r"offline[- /]online (?:correlation|metric)", r"online/offline metric",
    r"offline (?:evaluation|metrics|experimentation)", r"interleaving",
    r"personali[sz]ation (?:system|engine|model|pipeline|platform|layer|infrastructure)s?",
    r"improv(?:es?|ed|ing) relevance", r"surface (?:the )?(?:most )?relevant",
    r"revenue[- ]per[- ]search", r"time[- ]to[- ]shortlist",
]
MEDIUM_PHRASES = [  # tier B — production-ML adjacent, supporting only
    r"personali[sz]ation", r"\brelevance\b", r"click[- ]?through", r"\bctr\b",
    r"a/?b[- ]test", r"experiment(?:ation)? (?:framework|platform|environment)",
    r"feature (?:store|pipeline)s?", r"model (?:serving|deployment|monitoring|registry|drift)",
    r"(?:deployed|shipped|served|productioni[sz]ed)[^.]{0,40}model",
    r"model[^.]{0,30}(?:in|to|into) production", r"ml (?:pipeline|platform|infra(?:structure)?|system)s?",
    r"machine[- ]learning (?:model|pipeline|system)s?", r"trained (?:a |the )?model",
    r"model training", r"training pipeline", r"inference (?:latency|service|pipeline|server)",
    r"online inference", r"batch inference", r"real[- ]time (?:scoring|inference|predictions?)",
    r"fine[- ]tun(?:ed|ing)", r"\bpytorch\b", r"\btensorflow\b", r"\bxgboost\b", r"\blightgbm\b",
    r"gradient[- ]boost", r"natural language processing", r"\bnlp\b", r"text classification",
    r"sentiment analysis", r"document classification", r"named[- ]entity", r"topic model",
    r"\bdistilbert\b", r"hugging face", r"\btransformer[- ]based\b",
    r"elasticsearch", r"opensearch", r"\bsolr\b", r"\blucene\b", r"autocomplete", r"typeahead",
    r"churn (?:prediction|model)", r"propensity model", r"fraud[- ]detection", r"anomaly detection",
    r"forecasting model", r"prediction (?:service|pipeline|model|api)s?", r"predictive (?:model|features?)",
    r"data drift", r"drift (?:detection|monitoring)", r"retraining cadence", r"\bmlflow\b",
    r"\bkubeflow\b", r"\bbentoml\b", r"engagement signals", r"behavior(?:al)? signals",
]
WRAPPER_PHRASES = [  # negative — shallow genAI; search job_descriptions ONLY (summaries
                     # carry mass-produced 'AI-curious/self-learner' boilerplate, see Q4)
    r"\blangchain\b", r"\bllamaindex\b", r"llama[- ]index",
    r"openai(?:'s)? (?:api|apis|embeddings|gpt|models?|endpoints?)", r"open[- ]?ai api",
    r"gpt[- ]?(?:3(?:\.5)?|4|4o)\b", r"chatgpt", r"\bclaude\b", r"anthropic",
    r"prompt (?:engineering|templates?|chains?|tuning)", r"chat[- ]?bots?\b",
    r"conversational (?:bot|assistant|agent)s?", r"wrapper(?:s)? (?:around|over|on top of)",
    r"llm api", r"api calls? to (?:openai|gpt|llms?)", r"integrat(?:ed|ing) (?:openai|gpt|llms?|chatgpt)",
    r"side project", r"self[- ]learner", r"online courses",
    r"haven'?t done it in a professional capacity", r"experimented with chatgpt",
]
HEDGE_PHRASES = [  # negative — within-paragraph walk-backs (the corpus is full of them)
    r"didn'?t make it to production", r"wasn'?t? (?:deployed|shipped)", r"never (?:shipped|deployed)",
    r"my (?:own )?(?:modeling|technical) (?:work|depth)[^.]{0,30}(?:secondary|limited)",
    r"production deployment was handled by", r"not the model itself",
    r"more on the modeling side than the productioni[sz]ation",
    r"professional experience there is limited", r"haven'?t done much",
    r"my own technical depth in ai is limited", r"i wouldn'?t call myself an ml specialist",
    r"lighter weight than", r"adjacent ml exposure", r"some adjacent ml",
    r"limited (?:backend|production) exposure", r"not from[- ]scratch",
]
RESEARCH_PHRASES = [  # negative — academic-only careers (NOTE: absent from this corpus)
    r"\bpostdoc(?:toral)?\b", r"\bphd\b", r"\bdissertation\b", r"\bthesis\b",
    r"publish(?:ed)? (?:papers?|research|in)", r"peer[- ]review", r"first[- ]author",
    r"\bneurips\b", r"\bicml\b", r"\biclr\b", r"\bcvpr\b", r"\bemnlp\b", r"\bsigir\b",
    r"research (?:paper|publication|grant|fellowship)s?", r"h[- ]index", r"academic research",
    r"research lab", r"\blecturer\b", r"\bprofessor\b", r"research[- ]only",
]
PRODUCTION_PHRASES = [  # positive — shipped-to-production evidence
    r"\bproduction\b", r"\bdeployed?\b", r"\bshipp?(?:ed|ing)\b", r"\blaunch(?:ed)?\b",
    r"\bserving\b", r"\blatency\b", r"\bp95\b", r"\bthroughput\b", r"\buptime\b", r"on[- ]call",
    r"\brollout\b", r"\brollback\b", r"a/?b[- ]test", r"\bmonitoring\b", r"incident",
    r"million(?:s)? of (?:users|requests|queries|documents|items)", r"\d+m\+ (?:users|queries|items|documents)",
    r"billions of documents", r"index (?:refresh|versioning)",
]

AI_SKILLS = {  # AI tokens in the fixed 133-skill vocabulary (for the zero-AI-skill test)
    "Hugging Face Transformers", "LangChain", "Information Retrieval", "LLMs",
    "Recommendation Systems", "Semantic Search", "Sentence Transformers", "Embeddings",
    "Vector Search", "Prompt Engineering", "Pinecone", "FAISS", "RAG", "Fine-tuning LLMs",
    "YOLO", "GANs", "Feature Engineering", "OpenCV", "ASR", "Image Classification",
    "Computer Vision", "Speech Recognition", "CNN", "Kubeflow", "MLOps", "BentoML",
    "Data Science", "Reinforcement Learning", "Object Detection", "Diffusion Models",
    "MLflow", "Time Series", "Weights & Biases", "Forecasting", "TTS", "Statistical Modeling",
    "QLoRA", "pgvector", "Weaviate", "Milvus", "Learning to Rank", "BM25", "TensorFlow",
    "Qdrant", "PyTorch", "PEFT", "LoRA", "NLP", "Machine Learning", "Deep Learning",
    "Haystack", "LlamaIndex", "scikit-learn", "OpenSearch", "Information Retrieval Systems",
}
GENAI_WRAPPER_SKILLS = {"LangChain", "RAG", "Prompt Engineering", "LlamaIndex"}
CONSULTING_FIRMS = {"TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
                    "HCL", "Tech Mahindra", "Mphasis", "Mindtree"}

AI_TITLE_RE = re.compile(
    r"\bml\b|machine learning|\bai\b|artificial intelligence|data scien|deep learning|nlp|"
    r"computer vision|recommendation|search engineer|applied scien|applied ml|mlops|\(ml\)", re.I)
SWE_TITLES = {
    "Software Engineer", "Senior Software Engineer", "Full Stack Developer", "Backend Engineer",
    "Data Engineer", "Senior Data Engineer", "Analytics Engineer", "Frontend Engineer",
    "Cloud Engineer", "DevOps Engineer", "Java Developer", ".NET Developer",
    "Mobile Developer", "QA Engineer", "Data Analyst",
}

BOILER_AI_CURIOUS = "curious about how AI tools could augment my work"
BOILER_SELF_LEARNER = "self-learner level"
BOILER_STUFFER = "online courses on RAG and vector databases"  # keyword-stuffer summary

def compile_all(ps):
    return [re.compile(p, re.I) for p in ps]

STRONG_RE, MEDIUM_RE = compile_all(STRONG_PHRASES), compile_all(MEDIUM_PHRASES)
WRAPPER_RE, HEDGE_RE = compile_all(WRAPPER_PHRASES), compile_all(HEDGE_PHRASES)
RESEARCH_RE, PROD_RE = compile_all(RESEARCH_PHRASES), compile_all(PRODUCTION_PHRASES)

def hits(text, regs):
    return sum(1 for r in regs if r.search(text))

def para_meta(p):
    """Prefix-tolerant lookup into PARA_GRADES (keys are ~60-char prefixes)."""
    for k, v in PARA_GRADES.items():
        if p.startswith(k[:48]):
            return v
    return None

def main():
    df = pd.read_parquet(PARQUET)
    n = len(df)
    print(f"rows: {n}")

    # ---------- paragraph pool census + per-candidate fingerprint ----------
    pool = collections.Counter()
    cand_paras = []
    for s in df["job_descriptions"].fillna(""):
        ids = []
        for p in s.split("\n"):
            p = p.strip()
            if not p:
                continue
            pool[p] += 1
            meta = para_meta(p)
            ids.append(meta[0] if meta else "P??")
        cand_paras.append(ids)
    print(f"paragraph instances: {sum(pool.values())}, distinct paragraphs: {len(pool)}")
    unk = [p for p in pool if para_meta(p) is None]
    print(f"paragraphs not in hand-grade map: {len(unk)} (must be 0)")

    df["_paras"] = cand_paras
    id2grade = {v[0]: v[1] for v in PARA_GRADES.values()}
    df["_best_grade"] = df["_paras"].map(lambda ids: max((id2grade.get(i, 0) for i in ids), default=0))
    df["_n_grade5"] = df["_paras"].map(lambda ids: sum(1 for i in ids if id2grade.get(i, 0) == 5))
    df["_has_plain"] = df["_paras"].map(lambda ids: any(i in PLAIN_IDS for i in ids))

    # ---------- phrase counts: run regexes on the 44 paragraphs + distinct summaries ----------
    para_feat = {p: dict(strong=hits(p, STRONG_RE), medium=hits(p, MEDIUM_RE),
                         wrapper=hits(p, WRAPPER_RE), hedge=hits(p, HEDGE_RE),
                         research=hits(p, RESEARCH_RE), prod=hits(p, PROD_RE)) for p in pool}
    print("\n---- phrase-list validation against the 44-paragraph pool ----")
    print(f"{'id':4} {'grade':5} {'label':22} {'n':>6} strong med wrap hedge prod")
    for p, c in sorted(pool.items(), key=lambda x: -x[1]):
        pid, g, lab = para_meta(p)
        f = para_feat[p]
        print(f"{pid:4} {g:5} {lab:22} {c:6} {f['strong']:6} {f['medium']:3} "
              f"{f['wrapper']:4} {f['hedge']:5} {f['prod']:4}")
    g5_no_strong = [para_meta(p)[0] for p in pool
                    if para_meta(p)[1] == 5 and para_feat[p]["strong"] == 0]
    print(f"grade-5 paragraphs MISSED by STRONG list: {g5_no_strong} (target: empty)")
    g0_strong = [para_meta(p)[0] for p in pool
                 if para_meta(p)[1] == 0 and para_feat[p]["strong"] > 0]
    print(f"grade-0 paragraphs hit by STRONG list (false pos): {g0_strong} (target: empty)")

    def cand_count(key):
        m = {p: para_feat[p][key] for p in pool}
        per_para = {para_meta(p)[0]: m[p] for p in pool}
        return df["_paras"].map(lambda ids: sum(per_para.get(i, 0) for i in ids))
    df["_strong"] = cand_count("strong")
    df["_medium"] = cand_count("medium")
    df["_wrapper_desc"] = cand_count("wrapper")
    df["_hedge"] = cand_count("hedge")
    df["_prod"] = cand_count("prod")

    # summaries: evaluate on distinct values only
    summ = df["summary"].fillna("")
    distinct_summ = pd.Series(summ.unique())
    s_strong = {s: hits(s, STRONG_RE) for s in distinct_summ}
    df["_strong_sum"] = summ.map(s_strong)
    df["_boiler_curious"] = summ.str.contains(BOILER_AI_CURIOUS, case=False, regex=False)
    df["_boiler_selflearner"] = summ.str.contains(BOILER_SELF_LEARNER, case=False, regex=False)
    df["_strong_any"] = df["_strong"] + df["_strong_sum"]

    def title_class(t):
        t = t or ""
        if AI_TITLE_RE.search(t):
            return "AI-obvious"
        if t in SWE_TITLES:
            return "SWE-non-obvious"
        return "non-engineering"
    df["_tclass"] = df["current_title"].map(title_class)
    skills = df["skill_names"].fillna("").map(lambda s: set(s.split(" | ")) - {""})
    df["_n_ai_skills"] = skills.map(lambda s: len(s & AI_SKILLS))

    # ---------------- Q1 ----------------
    print("\n================ Q1: plain-language relevance detector ================")
    for thr in (1, 2, 3):
        m = df["_strong"] >= thr
        print(f"desc strong>={thr}: {m.sum()} ({m.mean()*100:.2f}%) | "
              f"{dict(df.loc[m, '_tclass'].value_counts())}")
    print(f"summary-only strong (strong_sum>=1 & desc strong==0): "
          f"{((df['_strong_sum']>=1)&(df['_strong']==0)).sum()}")
    m1 = df["_strong"] >= 1
    print("\ncurrent_title of desc-strong>=1 (all):")
    print(df.loc[m1, "current_title"].value_counts().to_string())
    swe1 = df[m1 & (df["_tclass"] == "SWE-non-obvious")]
    print(f"\nstrong>=1 with NON-obvious SWE title: {len(swe1)}")
    print(swe1["current_title"].value_counts().to_string())
    print(f"strong>=1 with non-engineering title: "
          f"{(m1 & (df['_tclass']=='non-engineering')).sum()}")
    print(f"\nbest_grade>=4 (real tier-4/5 paragraph holders): {(df['_best_grade']>=4).sum()}")
    print(f"best_grade==5: {(df['_best_grade']==5).sum()}")
    print(f"  of which non-AI-obvious title: "
          f"{((df['_best_grade']==5)&(df['_tclass']!='AI-obvious')).sum()}")
    print(f"holders of PLAIN paragraphs P39-P43: {df['_has_plain'].sum()}")
    print("plain-paragraph holders by title:")
    print(df.loc[df["_has_plain"], "current_title"].value_counts().to_string())

    # ---------------- Q2 ----------------
    print("\n================ Q2: hidden gems (strong text + ZERO AI skill_names) ================")
    for thr in (1, 2):
        gem = (df["_strong"] >= thr) & (df["_n_ai_skills"] == 0)
        print(f"desc strong>={thr} & 0 AI skills: {gem.sum()}")
    gem5 = (df["_best_grade"] == 5) & (df["_n_ai_skills"] == 0)
    print(f"best_grade==5 & 0 AI skills: {gem5.sum()}")
    gem45 = (df["_best_grade"] >= 4) & (df["_n_ai_skills"] == 0)
    print(f"best_grade>=4 & 0 AI skills: {gem45.sum()}")
    plain_noai = df["_has_plain"] & (df["_n_ai_skills"] == 0)
    print(f"PLAIN paragraph & 0 AI skills: {plain_noai.sum()}")
    print(f"PLAIN paragraph & <=2 AI skills: {(df['_has_plain']&(df['_n_ai_skills']<=2)).sum()}")
    print("\nAI-skill count distribution among best_grade==5 holders:")
    print(df.loc[df["_best_grade"] == 5, "_n_ai_skills"].value_counts().sort_index().to_string())
    gems = df[gem45 | plain_noai].copy().sort_values(["_has_plain", "_strong"], ascending=False)
    print(f"\nhidden-gem population (grade>=4 & 0 AI skills, or plain & 0 AI skills): {len(gems)}")
    print(f"  yoe 5-9: {gems['yoe'].between(5,9).sum()}; median yoe {gems['yoe'].median():.1f}")
    print(f"  India: {(gems['country']=='India').sum()}/{len(gems)}")
    print(f"  median days_since_active: {gems['days_since_active'].median():.0f}")
    print(f"  median sig_response_rate: {gems['sig_response_rate'].median()}")
    print("\nup to 30 hidden-gem ids:")
    print(gems[["candidate_id", "current_title", "yoe", "country", "_paras", "_n_ai_skills",
                "_strong", "days_since_active"]].head(30).to_string(index=False))

    # ---------------- Q3 ----------------
    print("\n================ Q3: hidden-gem full profiles from raw JSONL ================")
    plain_holders = df[df["_has_plain"]].sort_values("_n_ai_skills")
    pick = list(plain_holders["candidate_id"].head(8))
    pick += list(gems.loc[~gems["candidate_id"].isin(pick), "candidate_id"].head(6))
    pick = pick[:12]
    pickset, found = set(pick), {}
    cid_re = re.compile(r'"candidate_id"\s*:\s*"(CAND_\d+)"')
    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            m = cid_re.search(line)
            if not m or m.group(1) not in pickset:
                continue
            found[m.group(1)] = json.loads(line)
            if len(found) == len(pickset):
                break
    for cid in pick:
        r = found.get(cid)
        if not r:
            print(f"-- {cid}: NOT FOUND in JSONL")
            continue
        p = r.get("profile") or {}
        print("\n" + "=" * 90)
        print(f"{cid} | {p.get('anonymized_name')} | {p.get('headline')}")
        print(f"loc: {p.get('location')}, {p.get('country')} | yoe={p.get('years_of_experience')} | "
              f"current: {p.get('current_title')} @ {p.get('current_company')} "
              f"({p.get('current_industry')}, size {p.get('current_company_size')})")
        print(f"summary: {(p.get('summary') or '')[:330]}")
        for j in (r.get("career_history") or []):
            print(f"  JOB: {j.get('title')} @ {j.get('company')} ({j.get('start_date')} -> "
                  f"{j.get('end_date')}, {j.get('duration_months')}mo, ind={j.get('industry')})")
            print(f"       {(j.get('description') or '')[:240]}")
        sk = [f"{s.get('name')}({s.get('proficiency')},{s.get('duration_months')}mo)"
              for s in (r.get("skills") or [])]
        print(f"  SKILLS: {sk}")
        sig = r.get("redrob_signals") or {}
        print(f"  SIGNALS: resp_rate={sig.get('recruiter_response_rate')}, "
              f"last_active={sig.get('last_active_date')}, notice={sig.get('notice_period_days')}d, "
              f"relocate={sig.get('willing_to_relocate')}, open={sig.get('open_to_work_flag')}, "
              f"github={sig.get('github_activity_score')}, "
              f"salary_lpa={sig.get('expected_salary_range_inr_lpa')}")

    # ---------------- Q4 ----------------
    print("\n================ Q4: LangChain-wrapper / shallow-genAI anti-pattern ================")
    print(f"summary boilerplate 'AI-curious (ChatGPT productivity)': {df['_boiler_curious'].sum()}")
    print(f"summary boilerplate 'self-learner APIs + RAG side project, not professional': "
          f"{df['_boiler_selflearner'].sum()}")
    df["_boiler_stuffer"] = summ.str.contains(BOILER_STUFFER, case=False, regex=False)
    print(f"summary boilerplate STUFFER 'online courses on RAG and vector databases...': "
          f"{df['_boiler_stuffer'].sum()} | title class: "
          f"{dict(df.loc[df['_boiler_stuffer'],'_tclass'].value_counts())} | "
          f"with 4+ AI skills: {(df['_boiler_stuffer'] & (df['_n_ai_skills']>=4)).sum()} | "
          f"best_grade>0: {(df['_boiler_stuffer'] & (df['_best_grade']>0)).sum()}")
    print(f"  boilerplate by title class: curious={dict(df.loc[df['_boiler_curious'],'_tclass'].value_counts())}")
    print(f"  self-learner={dict(df.loc[df['_boiler_selflearner'],'_tclass'].value_counts())}")
    flashy = df["_tclass"] == "AI-obvious"
    print(f"\nAI-obvious current titles: {flashy.sum()}")
    print("best paragraph grade distribution for AI-obvious titles:")
    print(df.loc[flashy, "_best_grade"].value_counts().sort_index().to_string())
    shallow = flashy & (df["_best_grade"] <= 2)
    print(f"AI title but NO ML paragraph above grade 2 (title-evidence mismatch): {shallow.sum()}")
    def only_p30(ids):
        g35 = [i for i in ids if id2grade.get(i, 0) >= 3]
        return bool(g35) and all(i == "P30" for i in g35)
    p30_only = flashy & df["_paras"].map(only_p30)
    print(f"AI title whose ONLY >=grade-3 evidence is P30 (RAG chatbot): {p30_only.sum()}")
    cv_only = flashy & df["_paras"].map(
        lambda ids: any(i == "P23" for i in ids)
        and all(id2grade.get(i, 0) <= 2 for i in ids))
    print(f"AI title, CV-only evidence (P23, nothing >grade 2): {cv_only.sum()}")
    # genai skill months from skills_json
    def genai_months(sj):
        try:
            arr = json.loads(sj)
        except Exception:
            return -1
        ms = [s.get("duration_months") or 0 for s in arr if s.get("name") in GENAI_WRAPPER_SKILLS]
        return max(ms) if ms else -1
    df["_genai_mo"] = df["skills_json"].map(genai_months)
    wrap12 = flashy & (df["_genai_mo"].between(0, 11)) & (df["_best_grade"] <= 3)
    print(f"AI title + genai-wrapper skill <12mo + no grade-4/5 paragraph: {wrap12.sum()}")
    sample = df[shallow].head(12)
    print("\nsample title-evidence-mismatch ids:")
    print(sample[["candidate_id", "current_title", "yoe", "_paras", "_best_grade",
                  "_genai_mo"]].to_string(index=False))

    # ---------------- Q5 ----------------
    print("\n================ Q5: pure-research anti-pattern ================")
    acad = re.compile(r"universit|institute|academy|college|iisc|\blabs?\b|research (?:lab|center|centre)", re.I)
    comp_acad = df["job_companies"].fillna("").map(lambda s: bool(acad.search(s)))
    print(f"candidates with any academic-looking employer: {comp_acad.sum()}")
    res_title = df["job_titles"].fillna("").str.contains("research", case=False)
    print(f"candidates with any 'Research' job title: {res_title.sum()} "
          f"(all are 'AI Research Engineer' — an applied title here)")
    text_research = (df["_strong"] * 0 + df["job_descriptions"].fillna("")
                     .str.contains(r"\bphd\b|\bthesis\b|publish|peer.review|postdoc", case=False)).sum()
    print(f"candidates with academic-research language in descriptions: {text_research}")
    print(f"candidates with >=1 RESEARCH_PHRASES hit (desc): "
          f"{(cand_count('research')>=1).sum()}")
    print(f"hedge-language candidates (>=1 hedge phrase): {(df['_hedge']>=1).sum()}; "
          f">=2: {(df['_hedge']>=2).sum()}")
    are = df[df["current_title"] == "AI Research Engineer"]
    print(f"'AI Research Engineer' current title: {len(are)}; their best-grade dist: "
          f"{dict(are['_best_grade'].value_counts().sort_index())}")

    # ---------------- scorer crosstabs ----------------
    print("\n================ scorer-design crosstabs ================")
    print("best_grade distribution (all 100k):")
    print(df["_best_grade"].value_counts().sort_index().to_string())
    print("\nAI-skill bins vs best_grade:")
    ab = pd.cut(df["_n_ai_skills"], [-1, 0, 3, 100], labels=["0", "1-3", "4+"])
    print(pd.crosstab(ab, df["_best_grade"]).to_string())
    stuf = (df["_n_ai_skills"] >= 4) & (df["_best_grade"] == 0)
    print(f"\nkeyword stuffers (4+ AI skills, best_grade==0): {stuf.sum()}; by title class: "
          f"{dict(df.loc[stuf,'_tclass'].value_counts())}")
    g45 = df["_best_grade"] >= 4
    print(f"\ntier-4/5 paragraph holders total: {g45.sum()}  -> top-100 must come from here")
    print(f"  with consulting-only careers: "
          f"{(g45 & df['job_companies'].fillna('').map(lambda s: bool(s) and set(s.split(' | ')) <= CONSULTING_FIRMS)).sum()}")
    print(f"  India-based: {(g45 & (df['country']=='India')).sum()}")
    print(f"  yoe 5-9: {(g45 & df['yoe'].between(5,9)).sum()}")
    print(f"  grade5 & yoe5-9 & India: "
          f"{((df['_best_grade']==5) & df['yoe'].between(5,9) & (df['country']=='India')).sum()}")

    # ---------------- follow-ups ----------------
    print("\n================ follow-ups ================")
    # (a) summary-only strong: who are they?
    so = (df["_strong_sum"] >= 1) & (df["_strong"] == 0)
    print(f"summary-only strong: {so.sum()} | title class: "
          f"{dict(df.loc[so,'_tclass'].value_counts())} | best_grade dist: "
          f"{dict(df.loc[so,'_best_grade'].value_counts().sort_index())}")
    ex = df.loc[so, "summary"].iloc[0]
    print(f"  example summary: {ex[:220]}")

    # (b) plain-language SUMMARY template
    plain_sum = df["summary"].fillna("").str.contains(
        "connect users with relevant information at scale", case=False, regex=False)
    print(f"\nplain summary template 'connect users with relevant information at scale': "
          f"{plain_sum.sum()} candidates; their best_grade dist: "
          f"{dict(df.loc[plain_sum,'_best_grade'].value_counts().sort_index())}")

    # (c) paraphrased skill vocabulary (only plain-language profiles carry these)
    PARA_SKILLS = {"Search Infrastructure", "Model Adaptation", "Indexing Algorithms",
                   "Vector Representations", "Content Matching", "Text Encoders",
                   "Search & Discovery", "Ranking Systems", "Information Retrieval Systems",
                   "Open-source ML libraries", "Document Processing", "Search Backend",
                   "Workflow Orchestration", "Natural Language Processing"}
    skills2 = df["skill_names"].fillna("").map(lambda s: set(s.split(" | ")) - {""})
    df["_n_para_skills"] = skills2.map(lambda s: len(s & PARA_SKILLS))
    hp = df["_n_para_skills"] > 0
    print(f"\ncandidates with any PARAPHRASED skill: {hp.sum()}; "
          f"with >=3: {(df['_n_para_skills']>=3).sum()}")
    print(f"  their best_grade dist: {dict(df.loc[hp,'_best_grade'].value_counts().sort_index())}")
    print(f"  their current titles: {dict(df.loc[hp,'current_title'].value_counts())}")

    # (d) honeypot probe: skill duration_months vs career_span_months in the top pool
    def max_skill_mo(sj):
        try:
            arr = json.loads(sj)
        except Exception:
            return -1
        return max((s.get("duration_months") or 0) for s in arr) if arr else -1
    df["_max_skill_mo"] = df["skills_json"].map(max_skill_mo)
    impossible = df["_max_skill_mo"] > (df["career_span_months"] + 6)
    g45 = df["_best_grade"] >= 4
    print(f"\nskill-duration > career_span+6mo (impossible-history probe): "
          f"{impossible.sum()} overall; among grade>=4 pool: {(impossible & g45).sum()}; "
          f"among the 8 plain holders: {(impossible & df['_has_plain']).sum()}")
    print("plain holders, max skill months vs career span:")
    print(df.loc[df["_has_plain"], ["candidate_id", "yoe", "career_span_months",
                                    "_max_skill_mo", "_n_para_skills"]].to_string(index=False))
    print("\ngrade>=4 pool with impossible skill durations (first 15):")
    print(df.loc[impossible & g45, ["candidate_id", "current_title", "yoe",
                                    "career_span_months", "_max_skill_mo", "_best_grade"]]
          .head(15).to_string(index=False))

    # (e) which STRONG phrase fires on non-engineering SUMMARIES (false-positive control)
    so_noneng = so & (df["_tclass"] == "non-engineering")
    sample_sums = df.loc[so_noneng, "summary"].drop_duplicates().head(200)
    fp = collections.Counter()
    for s in sample_sums:
        for pat, rgx in zip(STRONG_PHRASES, STRONG_RE):
            if rgx.search(s):
                fp[pat] += 1
    print(f"\nSTRONG phrases firing on non-engineering summaries (200 distinct sampled): {dict(fp)}")

    # (f) tech-age violations in the grade>=4 pool (skill used longer than the tech exists;
    #     reference today = 2026-06; conservative month caps)
    TECH_AGE_CAP = {"QLoRA": 40, "LangChain": 46, "LlamaIndex": 46, "LoRA": 62, "PEFT": 52,
                    "pgvector": 64, "OpenSearch": 64, "Pinecone": 64, "Qdrant": 64,
                    "RAG": 44, "Prompt Engineering": 74, "Fine-tuning LLMs": 74}
    def tech_violation(sj):
        try:
            arr = json.loads(sj)
        except Exception:
            return []
        return [f"{s['name']}={s.get('duration_months')}mo" for s in arr
                if s.get("name") in TECH_AGE_CAP
                and (s.get("duration_months") or 0) > TECH_AGE_CAP[s["name"]]]
    df["_tech_viol"] = df["skills_json"].map(tech_violation)
    tv = df["_tech_viol"].map(len) > 0
    print(f"\ntech-age violations: overall {tv.sum()}; in grade>=4 pool {(tv & g45).sum()} "
          f"of {g45.sum()}; among 8 plain holders {(tv & df['_has_plain']).sum()}")
    print("plain holders' tech-age violations:")
    print(df.loc[df["_has_plain"], ["candidate_id", "_tech_viol"]].to_string(index=False))
    print("\ngrade-5 holders with tech-age violations (first 12):")
    print(df.loc[tv & (df["_best_grade"] == 5),
                 ["candidate_id", "current_title", "yoe", "_tech_viol"]]
          .head(12).to_string(index=False))
    clean5 = (df["_best_grade"] == 5) & ~tv & ~impossible & df["yoe"].between(5, 9) \
        & (df["country"] == "India")
    print(f"\nCLEAN grade-5 (no tech-age/career-span violation, yoe 5-9, India): {clean5.sum()}")

if __name__ == "__main__":
    main()
