# POOL CENSUS — Redrob ranker EDA
# Answers: title distribution, geography, yoe, JD funnel, industry/company,
# education, salary/notice segmentation. Deterministic; prints exact counts.
import re
import pandas as pd

pd.set_option("display.width", 200)

PARQUET = "c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons/redrob-ranker/data/candidates_flat.parquet"
df = pd.read_parquet(PARQUET)
N = len(df)
print(f"TOTAL CANDIDATES: {N}")

# ---------------------------------------------------------------- title sets
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
CV_ONLY_TITLES = {"Computer Vision Engineer"}  # JD: CV-only w/o NLP/IR is a disqualifier
DATA_ADJACENT_TITLES = {
    "Data Engineer", "Senior Data Engineer", "Analytics Engineer", "Data Analyst",
    "Backend Engineer",
}
GENERIC_SWE_TITLES = {
    "Software Engineer", "Senior Software Engineer", "Full Stack Developer",
    "Cloud Engineer", "Java Developer", ".NET Developer", "DevOps Engineer",
    "Mobile Developer", "Frontend Engineer", "QA Engineer",
}

def title_bucket(t):
    if t in CORE_AI_TITLES:
        return "1_core_ai_ml"
    if t in CV_ONLY_TITLES:
        return "2_cv_only"
    if t in DATA_ADJACENT_TITLES:
        return "3_data_adjacent"
    if t in GENERIC_SWE_TITLES:
        return "4_generic_swe"
    return "5_non_tech"

df["title_bucket"] = df.current_title.map(title_bucket)

print("\n================ Q1. CURRENT_TITLE DISTRIBUTION (all 47 titles) ================")
vc = df.current_title.value_counts()
for t, c in vc.items():
    print(f"{c:6d}  [{title_bucket(t)}]  {t}")
print("\nTitle bucket totals:")
print(df.title_bucket.value_counts().sort_index().to_string())
core_n = (df.title_bucket == "1_core_ai_ml").sum()
print(f"\nCore AI/ML titles: {core_n} ({100*core_n/N:.2f}% of pool)")
print(f"Core AI/ML + CV-only: {core_n + (df.title_bucket=='2_cv_only').sum()}")
print(f"Core+CV+data-adjacent: {(df.title_bucket.isin(['1_core_ai_ml','2_cv_only','3_data_adjacent'])).sum()}")

# ---------------------------------------------------------------- geography
print("\n================ Q2. GEOGRAPHY ================")
print(df.country.value_counts().to_string())
india = df[df.country == "India"]
print(f"\nIndia total: {len(india)}")
print("\nIndia locations:")
print(india.location.value_counts().to_string())

def city(loc, name):
    return india.location.str.startswith(name).sum()

pune = india.location.str.startswith("Pune").sum()
noida = india.location.str.startswith("Noida").sum()
hyd = india.location.str.startswith("Hyderabad").sum()
mum = india.location.str.startswith("Mumbai").sum()
delhi = india.location.str.startswith("Delhi").sum()
gurgaon = india.location.str.startswith("Gurgaon").sum()
blr = india.location.str.startswith("Bangalore").sum()
ncr = noida + delhi + gurgaon
print(f"\nPune: {pune} | Noida: {noida} | Pune+Noida: {pune+noida}")
print(f"Hyderabad: {hyd} | Mumbai: {mum} | Delhi: {delhi} | Gurgaon: {gurgaon} | Delhi NCR (Delhi+Noida+Gurgaon): {ncr}")
print(f"Bangalore: {blr}")
print(f"JD-preferred/welcome cities (Pune+Noida+Hyd+Mumbai+Delhi+Gurgaon): {pune+noida+hyd+mum+delhi+gurgaon}")
abroad = df[df.country != "India"]
print(f"\nAbroad total: {len(abroad)}")
print(f"Abroad & willing_to_relocate=True: {abroad.sig_willing_to_relocate.sum()} "
      f"({100*abroad.sig_willing_to_relocate.mean():.1f}%)")
print(f"India & willing_to_relocate=True: {india.sig_willing_to_relocate.sum()} "
      f"({100*india.sig_willing_to_relocate.mean():.1f}%)")
print("\nAbroad core-AI/ML titles by country:")
print(abroad[abroad.title_bucket == "1_core_ai_ml"].country.value_counts().to_string())
ab_ai = abroad[abroad.title_bucket == "1_core_ai_ml"]
print(f"Abroad core-AI/ML: {len(ab_ai)}; of those willing_to_relocate: {ab_ai.sig_willing_to_relocate.sum()}")

# ---------------------------------------------------------------- yoe
print("\n================ Q3. YOE DISTRIBUTION ================")
bins = [-1, 3, 5, 9, 12, 100]
labels = ["<3", "3-5", "5-9", "9-12", "12+"]
df["yoe_bucket"] = pd.cut(df.yoe, bins=bins, labels=labels)
print("Overall:")
print(df.yoe_bucket.value_counts().reindex(labels).to_string())
print(f"\nyoe stats: min={df.yoe.min():.1f} max={df.yoe.max():.1f} mean={df.yoe.mean():.2f} median={df.yoe.median():.1f}")
core = df[df.title_bucket == "1_core_ai_ml"]
print(f"\nWithin core AI/ML titles (n={len(core)}):")
print(core.yoe_bucket.value_counts().reindex(labels).to_string())
print(f"Core AI/ML with yoe in [5,9]: {((core.yoe>=5)&(core.yoe<=9)).sum()}")
print(f"Core AI/ML with yoe in [4,10]: {((core.yoe>=4)&(core.yoe<=10)).sum()}")
print(f"Core AI/ML with yoe in [6,8] (JD 'ideal'): {((core.yoe>=6)&(core.yoe<=8)).sum()}")
corecv = df[df.title_bucket.isin(["1_core_ai_ml", "2_cv_only"])]
print(f"\nCore+CV (n={len(corecv)}):")
print(corecv.yoe_bucket.value_counts().reindex(labels).to_string())
dadj = df[df.title_bucket == "3_data_adjacent"]
print(f"\nData-adjacent (n={len(dadj)}):")
print(dadj.yoe_bucket.value_counts().reindex(labels).to_string())

# ---------------------------------------------------------------- funnel
print("\n================ Q4. THE FUNNEL ================")
# strict gate (a): core AI/ML title OR strongly-relevant headline keywords
HEAD_KW = re.compile(
    r"\b(machine learning|ml|nlp|llm|rag|embedding|embeddings|retrieval|vector|"
    r"ranking|recommendation|recommender|search|semantic|data scien\w*|deep learning|"
    r"transformer\w*|genai|generative ai)\b", re.I)
hl_hit = df.headline.fillna("").str.contains(HEAD_KW)
gate_a = df.title_bucket.eq("1_core_ai_ml") | hl_hit
print(f"headline keyword hits (whole pool): {hl_hit.sum()}")
print(f"(a) core AI/ML title OR relevant headline:        {gate_a.sum()}")
gate_b = gate_a & df.yoe.between(4, 10)
print(f"(b) + yoe in [4,10]:                              {gate_b.sum()}")
gate_c = gate_b & ((df.country == "India") | df.sig_willing_to_relocate)
print(f"(c) + India or willing_to_relocate:               {gate_c.sum()}")
gate_d = gate_c & (df.days_since_active <= 60) & (df.sig_response_rate >= 0.2)
print(f"(d) + active<=60d & response_rate>=0.2:           {gate_d.sum()}")
# sub-breakdown of gate_d
gd = df[gate_d]
print(f"    of (d): core-AI/ML title {gd.title_bucket.eq('1_core_ai_ml').sum()}, "
      f"in India {(gd.country=='India').sum()}, "
      f"in Pune/Noida {gd.location.str.startswith(('Pune','Noida')).sum()}, "
      f"yoe 5-9 {gd.yoe.between(5,9).sum()}")
print(f"    of (d) title-bucket mix: {gd.title_bucket.value_counts().sort_index().to_dict()}")

# how much do activity gates alone cut (base rates)
print(f"\nBase rates: days_since_active<=60: {(df.days_since_active<=60).sum()} "
      f"({100*(df.days_since_active<=60).mean():.1f}%); "
      f"response_rate>=0.2: {(df.sig_response_rate>=0.2).sum()} "
      f"({100*(df.sig_response_rate>=0.2).mean():.1f}%); both: "
      f"{((df.days_since_active<=60)&(df.sig_response_rate>=0.2)).sum()}")

# relaxed gate (a'): retrieval/ranking/recommendation/search/embedding/vector-db
# evidence anywhere in job_titles + headline + summary + job_descriptions
RELAX_KW = re.compile(
    r"\b(retrieval|ranking|recommendation systems?|recommendation engine|recommendation system|"
    r"recommender|search (engine|infra|ranking|relevance|system)|semantic search|"
    r"embedding|embeddings|vector (db|database|search|index)|learning.to.rank|ndcg|mrr|"
    r"two.tower|faiss|pinecone|weaviate|milvus|qdrant|elasticsearch|opensearch|"
    r"hybrid search|information retrieval|bm25|reranking|re.ranking)\b", re.I)
blob = (df.job_titles.fillna("") + " || " + df.headline.fillna("") + " || " +
        df.summary.fillna("") + " || " + df.job_descriptions.fillna(""))
relax_hit = blob.str.contains(RELAX_KW)
print(f"\nRELAXED funnel (career-text evidence of retrieval/ranking/search/recsys):")
ga2 = relax_hit
print(f"(a') text evidence anywhere:                      {ga2.sum()}")
print(f"     ... of those, core-AI/ML title: {(df[ga2].title_bucket=='1_core_ai_ml').sum()}; "
      f"non-tech title: {(df[ga2].title_bucket=='5_non_tech').sum()} (keyword-stuffer zone)")
gb2 = ga2 & df.yoe.between(4, 10)
print(f"(b') + yoe in [4,10]:                             {gb2.sum()}")
gc2 = gb2 & ((df.country == "India") | df.sig_willing_to_relocate)
print(f"(c') + India or willing_to_relocate:              {gc2.sum()}")
gd2 = gc2 & (df.days_since_active <= 60) & (df.sig_response_rate >= 0.2)
print(f"(d') + active<=60d & response_rate>=0.2:          {gd2.sum()}")
print(f"     of (d') title-bucket mix: {df[gd2].title_bucket.value_counts().sort_index().to_dict()}")

# union shortlist
union_d = gate_d | gd2
print(f"\nUNION of strict (d) and relaxed (d') shortlists:  {union_d.sum()}")

# keyword-stuffer check: relevant keywords in skills but non-tech title
SKILL_KW = re.compile(r"\b(rag|llm|embedding|vector|pinecone|faiss|langchain|pytorch|"
                      r"transformers|huggingface|mlops|nlp)\b", re.I)
skill_hit = df.skill_names.fillna("").str.contains(SKILL_KW)
stuffers = skill_hit & df.title_bucket.eq("5_non_tech")
print(f"\nAI keywords in SKILLS overall: {skill_hit.sum()}; "
      f"with NON-TECH title (keyword stuffers): {stuffers.sum()}")
print("  stuffer titles:", df[stuffers].current_title.value_counts().head(12).to_dict())
# do stuffers also leak into job_descriptions evidence?
print(f"  stuffers that ALSO pass relaxed career-text gate: {(stuffers & relax_hit).sum()}")

# ---------------------------------------------------------------- industry/company
print("\n================ Q5. INDUSTRY & COMPANY ================")
print("current_industry distribution:")
print(df.current_industry.value_counts().to_string())
print(f"\ndistinct current_company: {df.current_company.nunique()}")
print("Top 30 current_company:")
print(df.current_company.value_counts().head(30).to_string())

CONSULT = ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini"]
cur_consult = df.current_company.isin(CONSULT)
print(f"\nCurrently at JD-named consulting firm: {cur_consult.sum()} ({100*cur_consult.mean():.1f}%)")
print(df[cur_consult].current_company.value_counts().to_string())
print(f"Core AI/ML title AND currently at consulting firm: {(cur_consult & df.title_bucket.eq('1_core_ai_ml')).sum()}")

def consulting_only(jc):
    comps = [c.strip() for c in str(jc).split("|") if c.strip()]
    return len(comps) > 0 and all(c in CONSULT for c in comps)

co_only = df.job_companies.map(consulting_only)
print(f"\nConsulting-ONLY entire career (every job at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini): "
      f"{co_only.sum()} ({100*co_only.mean():.1f}%)")
print(f"Consulting-only AND core AI/ML title: {(co_only & df.title_bucket.eq('1_core_ai_ml')).sum()}")
print(f"Consulting-only within strict funnel (d): {(co_only & gate_d).sum()}")
print(f"Consulting-only within relaxed funnel (d'): {(co_only & gd2).sum()}")
ever_consult = df.job_companies.fillna("").str.contains("|".join(CONSULT))
print(f"EVER worked at one of the six firms: {ever_consult.sum()} ({100*ever_consult.mean():.1f}%)")

print("\ncurrent_company_size distribution:")
print(df.current_company_size.value_counts(dropna=False).to_string())
print("\ncore AI/ML by company size:")
print(core.current_company_size.value_counts(dropna=False).to_string())

# ---------------------------------------------------------------- education
print("\n================ Q6. EDUCATION ================")
print("best_edu_tier distribution (all):")
print(df.best_edu_tier.value_counts(dropna=False).sort_index().to_string())
print("\nbest_edu_tier within core AI/ML titles:")
print(core.best_edu_tier.value_counts(dropna=False).sort_index().to_string())
print("\nbest_edu_tier within strict funnel (d):")
print(gd.best_edu_tier.value_counts(dropna=False).sort_index().to_string())
xt = pd.crosstab(df.best_edu_tier, df.title_bucket)
print("\ncrosstab best_edu_tier x title_bucket:")
print(xt.to_string())
print("\nshare of core-AI/ML within each edu tier (%):")
print((100 * xt["1_core_ai_ml"] / xt.sum(axis=1)).round(2).to_string())

# ---------------------------------------------------------------- salary/notice
print("\n================ Q7. SALARY & NOTICE (AI-relevant pool) ================")
pool = df[df.title_bucket.isin(["1_core_ai_ml"])]
pool_in = pool[pool.country == "India"]
pool_ab = pool[pool.country != "India"]
print(f"core AI/ML: n={len(pool)} (India {len(pool_in)}, abroad {len(pool_ab)})")
for name, p in [("India core-AI/ML", pool_in), ("Abroad core-AI/ML", pool_ab)]:
    print(f"\n{name} (n={len(p)}):")
    print("  sig_salary_min quantiles:",
          p.sig_salary_min.quantile([.1, .25, .5, .75, .9]).round(1).to_dict())
    print("  sig_salary_max quantiles:",
          p.sig_salary_max.quantile([.1, .25, .5, .75, .9]).round(1).to_dict())
print("\nWhole-pool India salary_min quantiles:",
      india.sig_salary_min.quantile([.1, .5, .9]).round(1).to_dict())
print("Whole-pool abroad salary_min quantiles:",
      abroad.sig_salary_min.quantile([.1, .5, .9]).round(1).to_dict())
# salary vs yoe within India core pool
print("\nIndia core-AI/ML median salary_min by yoe bucket:")
print(pool_in.groupby(pool_in.yoe_bucket, observed=True).sig_salary_min.median().round(1).to_string())

print("\nNotice period (days) distribution — core AI/ML pool:")
print(pool.sig_notice_period_days.value_counts().sort_index().to_string())
print("\nNotice period — whole pool:")
print(df.sig_notice_period_days.value_counts().sort_index().to_string())
print(f"\ncore AI/ML notice<=30d: {(pool.sig_notice_period_days<=30).sum()} "
      f"({100*(pool.sig_notice_period_days<=30).mean():.1f}%)")
print(f"strict funnel (d) notice<=30d: {(gd.sig_notice_period_days<=30).sum()} of {len(gd)}")
print(f"\nstrict funnel (d) salary_min quantiles: "
      f"{gd.sig_salary_min.quantile([.1,.5,.9]).round(1).to_dict()}")
print(f"strict funnel (d) work_mode mix: {gd.sig_work_mode.value_counts().to_dict()}")
print(f"strict funnel (d) open_to_work: {gd.sig_open_to_work.sum()} of {len(gd)}")

# final tightest cut for intuition: funnel (d) + notice<=30
tight = gate_d & (df.sig_notice_period_days <= 30)
print(f"\nTIGHTEST: strict funnel (d) + notice<=30d: {tight.sum()}")
tight2 = tight & df.location.str.startswith(("Pune", "Noida", "Hyderabad", "Mumbai", "Delhi", "Gurgaon"))
print(f"TIGHTEST + JD-welcome city: {tight2.sum()}")
print("\nSample candidate_ids from tightest pool (first 15):")
print(df[tight2].candidate_id.head(15).tolist())

# ---------------------------------------------------------------- verification
print("\n================ Q8. VERIFICATION & LEAKAGE PROBES ================")
# (v1) who are the 215 non-tech inside strict funnel (d)? stuffers via headline?
leak = gate_d & df.title_bucket.eq("5_non_tech")
print(f"(v1) non-tech inside strict funnel (d): {leak.sum()}")
print("  their titles:", df[leak].current_title.value_counts().head(15).to_dict())
print("  sample headlines:")
for h in df[leak].headline.head(8):
    print("   -", h)
print(f"  how many of them are skill-stuffers (AI kw in skills): {(leak & skill_hit).sum()}")
print(f"  how many pass career-text relax gate: {(leak & relax_hit).sum()}")

# (v2) notice-period leakage: are {0,15,45} day values tier-conditioned?
print("\n(v2) notice period crosstab title_bucket x value:")
print(pd.crosstab(df.sig_notice_period_days, df.title_bucket).to_string())
odd_notice = df.sig_notice_period_days.isin([0, 15, 45])
print(f"odd notice values {{0,15,45}}: total {odd_notice.sum()}; "
      f"core-AI/ML {(odd_notice & df.title_bucket.eq('1_core_ai_ml')).sum()}; "
      f"cv_only {(odd_notice & df.title_bucket.eq('2_cv_only')).sum()}; "
      f"other {(odd_notice & df.title_bucket.isin(['3_data_adjacent','4_generic_swe','5_non_tech'])).sum()}")
nonai_odd = odd_notice & ~df.title_bucket.isin(["1_core_ai_ml", "2_cv_only"])
print(f"NON-core/cv with odd notice: {nonai_odd.sum()}")
if nonai_odd.sum():
    print(df[nonai_odd][["candidate_id", "current_title", "headline", "yoe",
                         "sig_notice_period_days"]].head(20).to_string())
    print("  of those, pass relax career-text gate:", (nonai_odd & relax_hit).sum())

# (v3) salary leakage: is high salary band exclusive to core pool?
print("\n(v3) median sig_salary_min by title bucket:")
print(df.groupby("title_bucket").sig_salary_min.median().round(1).to_string())
hi_sal = df.sig_salary_min >= 19
print(f"salary_min>=19: total {hi_sal.sum()}; by bucket: "
      f"{df[hi_sal].title_bucket.value_counts().sort_index().to_dict()}")
nonai_hi = hi_sal & df.title_bucket.isin(["4_generic_swe", "5_non_tech"])
print(f"generic-swe/non-tech with salary_min>=19: {nonai_hi.sum()}")
if nonai_hi.sum():
    print("  titles:", df[nonai_hi].current_title.value_counts().head(10).to_dict())
    print("  pass relax career-text gate:", (nonai_hi & relax_hit).sum())
    print("  sample ids:", df[nonai_hi & relax_hit].candidate_id.head(10).tolist()
          if (nonai_hi & relax_hit).sum() else df[nonai_hi].candidate_id.head(5).tolist())

# (v4) yoe oddities inside core AI pool
print("\n(v4) core AI/ML yoe>=12 (the 9-12 hole makes these suspicious):")
odd_yoe = core[core.yoe >= 12]
print(odd_yoe[["candidate_id", "current_title", "yoe", "n_jobs", "career_span_months",
               "total_job_months", "date_duration_mismatch", "n_expert_zero_months",
               "current_company"]].to_string())
print("\ncore AI/ML yoe<3 count:", (core.yoe < 3).sum(),
      "| titles:", core[core.yoe < 3].current_title.value_counts().to_dict())
print("core AI/ML yoe 9-12 strictly:", ((core.yoe > 9) & (core.yoe < 12)).sum())
print("core AI/ML max yoe below 12:", core[core.yoe < 12].yoe.max())

# (v5) activity/behavioral medians by bucket (tier-conditioned envelopes?)
print("\n(v5) behavioral medians by title bucket:")
beh = df.groupby("title_bucket").agg(
    med_days_active=("days_since_active", "median"),
    med_resp=("sig_response_rate", "median"),
    med_views30=("sig_profile_views_30d", "median"),
    med_saved30=("sig_saved_by_recruiters_30d", "median"),
    med_github=("sig_github_activity", "median"),
    open_to_work_pct=("sig_open_to_work", "mean"),
).round(3)
print(beh.to_string())

# (v6) clean funnel: drop non-tech stuffers from gate (a)
clean_a = df.title_bucket.isin(["1_core_ai_ml", "2_cv_only"]) | \
    (hl_hit & ~df.title_bucket.eq("5_non_tech"))
clean_b = clean_a & df.yoe.between(4, 10)
clean_c = clean_b & ((df.country == "India") | df.sig_willing_to_relocate)
clean_d = clean_c & (df.days_since_active <= 60) & (df.sig_response_rate >= 0.2)
print(f"\n(v6) CLEAN funnel (non-tech-titled excluded from gate a):")
print(f"(a) {clean_a.sum()}  (b) {clean_b.sum()}  (c) {clean_c.sum()}  (d) {clean_d.sum()}")
print(f"    clean (d) bucket mix: {df[clean_d].title_bucket.value_counts().sort_index().to_dict()}")
print(f"    clean (d) in Pune/Noida: {df[clean_d].location.str.startswith(('Pune','Noida')).sum()}")

# (v7) plain-language tier-5 hunt: career-text evidence OUTSIDE core titles
print("\n(v7) relaxed-gate (a') bucket mix:",
      df[relax_hit].title_bucket.value_counts().sort_index().to_dict())
hidden = relax_hit & ~df.title_bucket.isin(["1_core_ai_ml", "2_cv_only"])
print(f"career-text evidence with NON-AI title (hidden-gem zone): {hidden.sum()}")
print("  titles:", df[hidden].current_title.value_counts().to_dict())
print("  also missed by strict gate (a):", (hidden & ~gate_a).sum())
print("  sample:", df[hidden & ~gate_a][["candidate_id", "current_title", "yoe"]]
      .head(10).to_string() if (hidden & ~gate_a).sum() else "none")
# plainer-language probe: recsys/personalization phrasing in job_descriptions
PLAIN_KW = re.compile(r"(recommendation (system|engine)|personalization|search relevance|"
                      r"a/b test\w*|click.through|collaborative filtering|cold.start)", re.I)
plain_hit = df.job_descriptions.fillna("").str.contains(PLAIN_KW)
print(f"\nplain recsys/AB-test phrasing in job_descriptions: {plain_hit.sum()}; bucket mix:",
      df[plain_hit].title_bucket.value_counts().sort_index().to_dict())
plain_hidden = plain_hit & ~df.title_bucket.isin(["1_core_ai_ml", "2_cv_only"]) & ~gate_a
print(f"plain-phrasing hits invisible to strict gate (a): {plain_hidden.sum()}")
if plain_hidden.sum():
    print(df[plain_hidden][["candidate_id", "current_title", "yoe", "country"]].head(15).to_string())

# (v8) github_activity envelope: -1 sentinel by bucket
print("\n(v8) sig_github_activity == -1 (missing) rate by bucket:")
print(df.groupby("title_bucket").sig_github_activity.apply(
    lambda s: round(100 * (s == -1).mean(), 1)).to_string())
gh_pos = df.sig_github_activity > 0
print(f"github_activity>0: total {gh_pos.sum()}; bucket mix:",
      df[gh_pos].title_bucket.value_counts().sort_index().to_dict())

# (v9) core-AI yoe range sanity
print("\n(v9) core AI/ML yoe: min", core.yoe.min(), "max", core.yoe.max(),
      "| n with yoe<=9:", (core.yoe <= 9).sum(), "| n with yoe>9:", (core.yoe > 9).sum())
print("claimed-vs-span gap (yoe*12 - career_span_months) for core yoe>=12:")
g = core[core.yoe >= 12]
print(((g.yoe * 12) - g.career_span_months).round(0).tolist())

# (v10) which plain phrases discriminate? phrase x bucket counts
print("\n(v10) plain-phrase hit counts by bucket (job_descriptions):")
jd_txt = df.job_descriptions.fillna("")
for pat in [r"recommendation (system|engine)", r"personalization", r"search relevance",
            r"a/b test", r"click.through", r"collaborative filtering", r"cold.start",
            r"\bembedding", r"\bretrieval\b", r"\branking\b", r"vector", r"\bndcg\b",
            r"\bmrr\b", r"semantic search", r"\bbm25\b", r"fine.tun", r"\blora\b"]:
    hit = jd_txt.str.contains(pat, case=False, regex=True)
    mix = df[hit].title_bucket.value_counts().sort_index().to_dict()
    print(f"  {pat!r:>30}: total {hit.sum():5d}  {mix}")

# (v11) gate (a) bucket mix — size of headline-stuffer entry
print("\n(v11) strict gate (a) bucket mix:",
      df[gate_a].title_bucket.value_counts().sort_index().to_dict())
print("headline-kw hit by bucket:",
      df[hl_hit].title_bucket.value_counts().sort_index().to_dict())

# (v12) core AI/ML geography detail + clean funnel education
print("\n(v12) core AI/ML by city (India):")
core_in = core[core.country == "India"]
for c in ["Pune", "Noida", "Hyderabad", "Mumbai", "Delhi", "Gurgaon", "Bangalore"]:
    print(f"  {c}: {core_in.location.str.startswith(c).sum()}")
print(f"  core AI/ML India total: {len(core_in)}; "
      f"JD-welcome cities: {core_in.location.str.startswith(('Pune','Noida','Hyderabad','Mumbai','Delhi','Gurgaon')).sum()}")
print("clean funnel (d) edu tiers:",
      df[clean_d].best_edu_tier.value_counts().sort_index().to_dict())
print("clean funnel (d) notice<=30:", (df[clean_d].sig_notice_period_days <= 30).sum())
print("clean funnel (d) ids (first 10):", df[clean_d].candidate_id.head(10).tolist())
