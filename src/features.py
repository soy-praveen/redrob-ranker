"""Feature extraction: candidates.jsonl -> per-candidate feature table.

Reads the raw JSONL (optionally gzipped) in one streaming pass and derives
every feature the scorer needs. Text analysis is memoized over *distinct*
strings: the corpus is heavily templated (44 distinct job-description
paragraphs, ~3.2k distinct summaries in the released pool), so each regex
runs once per distinct text instead of once per candidate. On a fresh
corpus with no duplication the same code still works, just slower.

No network, no models, CPU-only. Reference date is fixed for reproducibility.
"""

import gzip
import json
import re
from datetime import date
from functools import lru_cache

import pandas as pd

REFERENCE_DATE = date(2026, 6, 10)

# --------------------------------------------------------------------------
# Title taxonomy (closed 46-title vocabulary in the released pool; regex
# fallback keeps classification sane if new titles appear in an eval set)
# --------------------------------------------------------------------------
NONTECH_TITLES = {
    "Business Analyst", "HR Manager", "Mechanical Engineer", "Accountant",
    "Project Manager", "Customer Support", "Operations Manager", "Content Writer",
    "Sales Executive", "Civil Engineer", "Graphic Designer", "Marketing Manager",
}
SWE_TITLES = {
    "Software Engineer", "Full Stack Developer", "Cloud Engineer", "Java Developer",
    ".NET Developer", "DevOps Engineer", "Mobile Developer", "Frontend Engineer",
    "QA Engineer",
}
DATA_TITLES = {
    "Analytics Engineer", "Data Engineer", "Data Analyst", "Backend Engineer",
    "Senior Data Engineer", "Senior Software Engineer",
}
CV_TITLES = {"Computer Vision Engineer"}
AI_TITLE_RE = re.compile(
    r"\bml\b|machine learning|\bai\b|artificial intelligence|data scien|deep learning"
    r"|\bnlp\b|recommendation|search engineer|applied scien|applied ml|mlops|\(ml\)",
    re.I,
)


def title_class(title):
    if title in NONTECH_TITLES:
        return "NONTECH"
    if title in SWE_TITLES:
        return "SWE"
    if title in DATA_TITLES:
        return "DATA"
    if title in CV_TITLES:
        return "CV"
    if AI_TITLE_RE.search(title or ""):
        return "AI"
    return "OTHER"


# --------------------------------------------------------------------------
# Summary template families (exact substrings of the generator's templates)
# --------------------------------------------------------------------------
SUMMARY_FAMILIES = [
    # (family, substring) — first match wins
    ("ELITE", "hands-on experience building production ML systems, "
              "with a focus on search, retrieval, and ranking"),
    ("PLAIN", "building systems that connect users with relevant information at scale"),
    ("SOLID", "experience building ML-powered features in production. Strong background "
              "in NLP, recommendation systems"),
    ("STUFFER", "online courses on RAG and vector databases"),
    ("SWE_DISCLAIMER", "haven't done it in a professional capacity"),
    ("AI_CURIOUS", "experimented with ChatGPT"),
]

# --------------------------------------------------------------------------
# Skill vocabularies
# --------------------------------------------------------------------------
JD_CORE_SKILLS = {  # the 14-skill pool keyword stuffers draw from
    "LangChain", "LLMs", "Information Retrieval", "Semantic Search",
    "Hugging Face Transformers", "Vector Search", "FAISS", "Pinecone",
    "Prompt Engineering", "Embeddings", "Sentence Transformers",
    "Recommendation Systems", "RAG", "Fine-tuning LLMs",
}
PARAPHRASED_SKILLS = {  # de-buzzworded vocabulary unique to the plain-language seniors
    "Search Infrastructure", "Model Adaptation", "Indexing Algorithms",
    "Vector Representations", "Content Matching", "Text Encoders",
    "Search & Discovery", "Ranking Systems", "Information Retrieval Systems",
    "Open-source ML libraries", "Document Processing", "Search Backend",
    "Workflow Orchestration", "Natural Language Processing",
}
AI_SKILLS = {
    "Hugging Face Transformers", "LangChain", "Information Retrieval", "LLMs",
    "Recommendation Systems", "Semantic Search", "Sentence Transformers", "Embeddings",
    "Vector Search", "Prompt Engineering", "Pinecone", "FAISS", "RAG", "Fine-tuning LLMs",
    "YOLO", "GANs", "OpenCV", "ASR", "Image Classification", "Computer Vision",
    "Speech Recognition", "CNN", "Kubeflow", "MLOps", "BentoML", "Reinforcement Learning",
    "Object Detection", "Diffusion Models", "MLflow", "Weights & Biases", "TTS",
    "QLoRA", "pgvector", "Weaviate", "Milvus", "Learning to Rank", "BM25",
    "Qdrant", "PEFT", "LoRA", "NLP", "Machine Learning", "Deep Learning",
    "Haystack", "LlamaIndex", "OpenSearch", "Information Retrieval Systems",
    "PyTorch", "TensorFlow", "scikit-learn",
}

# JD-relevant assessment topics (retrieval/LLM stack; the generator assigns these
# to ~3% of the pool vs ~11% for generic-ML topics)
JD_STACK_TOPICS = {
    "Information Retrieval", "PEFT", "Milvus", "pgvector", "PyTorch", "Qdrant",
    "Machine Learning", "QLoRA", "Sentence Transformers", "Learning to Rank",
    "Embeddings", "Vector Search", "TensorFlow", "RAG", "Recommendation Systems",
    "Pinecone", "LLMs", "Weaviate", "LoRA", "Hugging Face Transformers", "OpenSearch",
    "scikit-learn", "Haystack", "Prompt Engineering", "Python", "BM25", "NLP",
    "FAISS", "Semantic Search", "LangChain", "LlamaIndex", "Elasticsearch",
    "Deep Learning", "Fine-tuning LLMs",
}

CONSULTING_FIRMS = {"TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini"}

# --------------------------------------------------------------------------
# Location ladder (from the JD: Pune/Noida preferred; Hyd/Mum/Delhi NCR
# welcome; other India fine; abroad needs relocation, no visa sponsorship)
# --------------------------------------------------------------------------
CITY_TIERS = {
    "pune": 3, "noida": 3,
    "hyderabad": 2, "mumbai": 2, "delhi": 2, "gurgaon": 2,
}


def location_tier(location, country, willing_to_relocate):
    if (country or "").strip().lower() == "india":
        city = (location or "").split(",")[0].strip().lower()
        return CITY_TIERS.get(city, 1)
    return 0.5 if willing_to_relocate else 0.0


# --------------------------------------------------------------------------
# Evidence lexicons applied to job_descriptions (career history text only;
# summaries are template boilerplate and never earn positive evidence)
# --------------------------------------------------------------------------
STRONG_PHRASES = [
    r"recommendation (?:system|engine)s?", r"\brecsys\b", r"learning[- ]to[- ]rank",
    r"\bltr\b", r"rank(?:ing|er)\b", r"information retrieval", r"\bretrieval\b",
    r"semantic search", r"vector (?:search|database|db|index|store)s?", r"\bfaiss\b",
    r"\bpinecone\b", r"\bweaviate\b", r"\bqdrant\b", r"\bmilvus\b", r"\bpgvector\b",
    r"\bbm25\b", r"hybrid (?:search|retrieval)", r"dense retrieval",
    r"search (?:quality|infrastructure|platform|backend|stack)", r"search and discovery",
    r"embedding(?:s)?\b", r"two[- ]tower", r"bi[- ]encoder", r"cross[- ]encoder",
    r"dual encoder", r"sentence[- ]transformers?", r"\bsbert\b", r"\bbge\b",
    r"matching (?:engine|algorithm|system|model|pipeline|logic|layer|score)s?",
    r"candidate[- ]jd matching", r"(?:most )?relevant (?:matches|results|content)",
    r"\bndcg\b", r"\bmrr\b", r"recall@\w+", r"precision@\w+",
    r"offline (?:evaluation|metrics|experimentation)", r"interleaving",
    r"personali[sz]ation (?:system|engine|model|pipeline|platform|layer|infrastructure)s?",
    r"improv(?:es?|ed|ing) relevance", r"surface (?:the )?(?:most )?relevant",
    r"collaborative filtering", r"content[- ]based filtering", r"cold[- ]start",
    r"query understanding", r"click[- ]?through",
]
MEDIUM_PHRASES = [
    r"personali[sz]ation", r"\brelevance\b", r"\bctr\b", r"a/?b[- ]test",
    r"feature (?:store|pipeline)s?", r"model (?:serving|deployment|monitoring|registry|drift)",
    r"(?:deployed|shipped|served|productioni[sz]ed)[^.]{0,40}model",
    r"model[^.]{0,30}(?:in|to|into) production", r"ml (?:pipeline|platform|infra(?:structure)?|system)s?",
    r"machine[- ]learning (?:model|pipeline|system)s?", r"trained (?:a |the )?model",
    r"fine[- ]tun(?:ed|ing)", r"\bpytorch\b", r"\btensorflow\b", r"\bxgboost\b",
    r"\blightgbm\b", r"natural language processing", r"\bnlp\b", r"text classification",
    r"named[- ]entity", r"topic model", r"hugging face", r"\btransformer[- ]based\b",
    r"elasticsearch", r"opensearch", r"\bsolr\b", r"\blucene\b", r"autocomplete",
    r"inference (?:latency|service|pipeline|server)", r"real[- ]time (?:scoring|inference|predictions?)",
    r"churn (?:prediction|model)", r"anomaly detection", r"engagement signals",
    r"behavior(?:al)? signals",
]
HEDGE_PHRASES = [
    r"didn'?t make it to production", r"wasn'?t? (?:deployed|shipped)", r"never (?:shipped|deployed)",
    r"my (?:own )?(?:modeling|technical) (?:work|depth)[^.]{0,30}(?:secondary|limited)",
    r"production deployment was handled by", r"not the model itself",
    r"more on the modeling side than the productioni[sz]ation",
    r"professional experience there is limited", r"haven'?t done much",
    r"my own technical depth in ai is limited", r"i wouldn'?t call myself an ml specialist",
    r"lighter weight than", r"adjacent ml exposure", r"some adjacent ml",
    r"limited (?:backend|production) exposure", r"not from[- ]scratch",
]
WRAPPER_PHRASES = [
    r"\blangchain\b", r"\bllamaindex\b", r"openai(?:'s)? (?:api|apis|embeddings|gpt|models?)",
    r"gpt[- ]?(?:3(?:\.5)?|4|4o)\b", r"chatgpt", r"prompt (?:engineering|templates?|chains?)",
    r"chat[- ]?bots?\b", r"conversational (?:bot|assistant|agent)s?",
    r"wrapper(?:s)? (?:around|over|on top of)", r"llm api",
    r"integrat(?:ed|ing) (?:openai|gpt|llms?|chatgpt)",
]
PRODUCTION_PHRASES = [
    r"\bproduction\b", r"\bdeployed?\b", r"\bshipp?(?:ed|ing)\b", r"\blaunch(?:ed)?\b",
    r"\bserving\b", r"\blatency\b", r"\bp95\b", r"\bthroughput\b", r"on[- ]call",
    r"a/?b[- ]test", r"\bmonitoring\b",
    r"million(?:s)? of (?:users|requests|queries|documents|items)",
]

_LEX = {
    "strong": [re.compile(p, re.I) for p in STRONG_PHRASES],
    "medium": [re.compile(p, re.I) for p in MEDIUM_PHRASES],
    "hedge": [re.compile(p, re.I) for p in HEDGE_PHRASES],
    "wrapper": [re.compile(p, re.I) for p in WRAPPER_PHRASES],
    "production": [re.compile(p, re.I) for p in PRODUCTION_PHRASES],
}

SUMMARY_YOE_RE = re.compile(r"(\d+\.?\d*)\+? (?:years|yrs)(?: of| hands-on)? experience", re.I)


@lru_cache(maxsize=None)
def scan_text(text):
    """Count lexicon hits in one distinct text. Cached: the corpus is templated."""
    return {name: sum(1 for rx in rxs if rx.search(text)) for name, rxs in _LEX.items()}


@lru_cache(maxsize=None)
def summary_features(summary):
    fam = "OTHER"
    for family, sub in SUMMARY_FAMILIES:
        if sub in summary:
            fam = family
            break
    m = SUMMARY_YOE_RE.search(summary)
    return fam, (float(m.group(1)) if m else None)


# --------------------------------------------------------------------------
# Date helpers
# --------------------------------------------------------------------------
def parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def months_between(d1, d2):
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


# --------------------------------------------------------------------------
# Main extraction
# --------------------------------------------------------------------------
def iter_candidates(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_one(c):
    p = c.get("profile", {}) or {}
    jobs = c.get("career_history", []) or []
    skills = c.get("skills", []) or []
    certs = c.get("certifications", []) or []
    sig = c.get("redrob_signals", {}) or {}

    # ---- career dates / honeypot markers -------------------------------
    date_duration_mismatch = 0
    starts = []
    for j in jobs:
        s, e = parse_date(j.get("start_date")), parse_date(j.get("end_date"))
        if s:
            starts.append(s)
            span = months_between(s, e or REFERENCE_DATE)
            claimed = j.get("duration_months")
            if claimed is not None and abs(span - claimed) > 3:
                date_duration_mismatch += 1
    career_start = min(starts, default=None)
    career_span_months = months_between(career_start, REFERENCE_DATE) if career_start else 0

    yoe = float(p.get("years_of_experience") or 0)
    summary = p.get("summary") or ""
    family, summary_yoe = summary_features(summary)

    hp_expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
    )
    hp_yoe_span = yoe * 12 > career_span_months + 24
    hp_yoe_summary = summary_yoe is not None and abs(summary_yoe - yoe) > 1.5
    hp_future_cert = any((cc.get("year") or 0) > REFERENCE_DATE.year for cc in certs)

    # ---- text evidence (memoized over distinct paragraphs) --------------
    para_prefixes = []
    text_counts = {"strong": 0, "medium": 0, "hedge": 0, "wrapper": 0, "production": 0}
    for j in jobs:
        desc = (j.get("description") or "").strip()
        if not desc:
            continue
        para_prefixes.append(desc[:60])
        hits = scan_text(desc)
        for k in text_counts:
            text_counts[k] += hits[k]

    # ---- skills ----------------------------------------------------------
    jd_core = [s for s in skills if s.get("name") in JD_CORE_SKILLS]
    jd_core_durs = [s.get("duration_months") or 0 for s in jd_core]
    jd_core_ends = [s.get("endorsements") or 0 for s in jd_core]
    skillset_key = "|".join(sorted(s.get("name") or "" for s in skills))

    # ---- assessments -----------------------------------------------------
    assess = sig.get("skill_assessment_scores") or {}
    jd_stack = {k: v for k, v in assess.items() if k in JD_STACK_TOPICS}

    # ---- career structure ------------------------------------------------
    durations = [j.get("duration_months") or 0 for j in jobs]
    companies = [(j.get("company") or "").strip() for j in jobs]
    consulting_only = bool(companies) and all(co in CONSULTING_FIRMS for co in companies)

    last_active = parse_date(sig.get("last_active_date"))
    sal = sig.get("expected_salary_range_inr_lpa") or {}

    return {
        "candidate_id": c.get("candidate_id"),
        "name": p.get("anonymized_name"),
        "headline": p.get("headline"),
        "current_title": p.get("current_title"),
        "current_company": p.get("current_company"),
        "current_industry": p.get("current_industry"),
        "location": p.get("location"),
        "country": p.get("country"),
        "yoe": yoe,
        "title_class": title_class(p.get("current_title")),
        "family": family,
        "summary_yoe": summary_yoe,
        "career_span_months": career_span_months,
        "agreed_yoe": min(yoe, career_span_months / 12.0) if career_span_months else yoe,
        # honeypot markers (profile-internal impossibilities only)
        "hp_expert_zero": hp_expert_zero > 0,
        "hp_date_mismatch": date_duration_mismatch > 0,
        "hp_yoe_span": hp_yoe_span,
        "hp_yoe_summary": hp_yoe_summary,
        "hp_future_cert": hp_future_cert,
        # text evidence
        "para_prefixes": para_prefixes,
        "n_strong": text_counts["strong"],
        "n_medium": text_counts["medium"],
        "n_hedge": text_counts["hedge"],
        "n_wrapper": text_counts["wrapper"],
        "n_production": text_counts["production"],
        # skills
        "n_skills": len(skills),
        "n_jd_core": len(jd_core),
        "jd_core_mean_dur": (sum(jd_core_durs) / len(jd_core_durs)) if jd_core_durs else 0.0,
        "jd_core_mean_endorse": (sum(jd_core_ends) / len(jd_core_ends)) if jd_core_ends else 0.0,
        "n_paraphrased": sum(1 for s in skills if s.get("name") in PARAPHRASED_SKILLS),
        "n_ai_skills": sum(1 for s in skills if s.get("name") in AI_SKILLS),
        "skillset_key": skillset_key,
        # assessments
        "n_assessments": len(assess),
        "n_jd_stack_topics": len(jd_stack),
        "jd_stack_mean": (sum(jd_stack.values()) / len(jd_stack)) if jd_stack else None,
        # career structure
        "n_jobs": len(jobs),
        "avg_tenure_months": (sum(durations) / len(durations)) if durations else 0.0,
        "n_short_stints": sum(1 for d in durations if d < 18),
        "consulting_only": consulting_only,
        "current_consulting": (p.get("current_company") or "").strip() in CONSULTING_FIRMS,
        # logistics / signals
        "location_tier": location_tier(p.get("location"), p.get("country"),
                                       bool(sig.get("willing_to_relocate"))),
        "willing_to_relocate": bool(sig.get("willing_to_relocate")),
        "days_since_active": (REFERENCE_DATE - last_active).days if last_active else None,
        "response_rate": sig.get("recruiter_response_rate"),
        "notice_days": sig.get("notice_period_days"),
        "open_to_work": bool(sig.get("open_to_work_flag")),
        "profile_completeness": sig.get("profile_completeness_score"),
        "profile_views_30d": sig.get("profile_views_received_30d"),
        "search_appearance_30d": sig.get("search_appearance_30d"),
        "saved_by_recruiters_30d": sig.get("saved_by_recruiters_30d"),
        "github_activity": sig.get("github_activity_score"),
        "interview_completion": sig.get("interview_completion_rate"),
        "salary_min": sal.get("min"),
        "salary_max": sal.get("max"),
        "endorsements_received": sig.get("endorsements_received"),
    }


def build_features(candidates_path):
    rows = [extract_one(c) for c in iter_candidates(candidates_path)]
    return pd.DataFrame(rows)
