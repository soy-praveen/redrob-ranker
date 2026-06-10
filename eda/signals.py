# Behavioral signals EDA — redrob ranker
# Question: do redrob_signals look tier-conditioned (ground-truth leakage) or independent noise?
# Run: PYTHONIOENCODING=utf-8 python eda/signals.py
import json
import os
import re
import warnings
import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
np.random.seed(0)

PARQUET = "c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons/redrob-ranker/data/candidates_flat.parquet"
df = pd.read_parquet(PARQUET)
N = len(df)
print(f"rows={N}")

NUMSIGS = [
    "sig_profile_completeness", "sig_profile_views_30d", "sig_applications_30d",
    "sig_response_rate", "sig_avg_response_hours", "sig_n_assessments",
    "sig_assessment_mean", "sig_connection_count", "sig_endorsements_received",
    "sig_notice_period_days", "sig_salary_min", "sig_salary_max",
    "sig_github_activity", "sig_search_appearance_30d", "sig_saved_by_recruiters_30d",
    "sig_interview_completion", "sig_offer_acceptance", "days_since_active",
]
BOOLSIGS = ["sig_open_to_work", "sig_willing_to_relocate", "sig_verified_email",
            "sig_verified_phone", "sig_linkedin_connected", "signup_after_active"]


def hist(s, bins, label=None):
    c = pd.cut(s, bins=bins, include_lowest=True).value_counts().sort_index()
    print(f"  hist {label or s.name}:")
    for iv, n in c.items():
        print(f"    {str(iv):>22} {n:>7} {100*n/len(s):5.1f}%")


print("\n" + "=" * 80)
print("S1. DISTRIBUTIONS OF EVERY SIGNAL")
print("=" * 80)
d = df[NUMSIGS].describe().T
d["nunique"] = [df[c].nunique() for c in NUMSIGS]
print(d[["mean", "std", "min", "25%", "50%", "75%", "max", "nunique"]].round(3))

print("\nbool/categorical base rates:")
for c in BOOLSIGS:
    print(f"  {c:28s} True={df[c].mean()*100:5.2f}%")
print(df["sig_work_mode"].value_counts().to_string())

print("\nsentinel / boundary masses:")
print(f"  sig_github_activity == -1 (no GitHub): {(df.sig_github_activity==-1).mean()*100:.2f}%")
print(f"  sig_interview_completion == -1:        {(df.sig_interview_completion==-1).mean()*100:.2f}%")
print(f"  sig_offer_acceptance == -1:            {(df.sig_offer_acceptance==-1).mean()*100:.2f}%")
print(f"  sig_assessment_mean NaN (no tests):    {df.sig_assessment_mean.isna().mean()*100:.2f}%")
print(f"  sig_response_rate == 0:  {(df.sig_response_rate==0).mean()*100:.2f}%   ==1: {(df.sig_response_rate==1).mean()*100:.2f}%")
print(f"  sig_profile_completeness ==1: {(df.sig_profile_completeness==1).mean()*100:.2f}%  ==min({df.sig_profile_completeness.min():.2f}): {(df.sig_profile_completeness==df.sig_profile_completeness.min()).mean()*100:.2f}%")
print(f"  sig_notice_period_days uniques: {sorted(df.sig_notice_period_days.unique())}")

hist(df.sig_profile_completeness, [0, 30, 40, 50, 60, 70, 80, 90, 100])
hist(df.sig_response_rate, [-0.001, 0, .1, .25, .5, .75, .9, 1.0])
hist(df.sig_github_activity, [-1.001, -0.999, 0, 10, 25, 50, 75, 100])
hist(df.sig_avg_response_hours, [0, 2, 6, 12, 24, 48, 96, 200, 1000])
hist(df.sig_connection_count, [-1, 0, 50, 150, 300, 500, 1000, 3000, 50000])
hist(df.sig_profile_views_30d, [-1, 0, 5, 15, 30, 60, 120, 10000])
hist(df.sig_applications_30d, [-1, 0, 2, 5, 10, 20, 50, 1000])
hist(df.sig_search_appearance_30d, [-1, 0, 10, 30, 60, 120, 250, 100000])
hist(df.sig_saved_by_recruiters_30d, [-1, 0, 1, 3, 6, 12, 25, 1000])
hist(df.sig_salary_min, [0, 5, 10, 15, 20, 30, 50, 100, 10000], "salary_min (LPA)")
hist(df.sig_salary_max, [0, 5, 10, 15, 20, 30, 50, 100, 10000], "salary_max (LPA)")
hist(df.sig_interview_completion[df.sig_interview_completion >= 0], [0, .2, .4, .6, .8, .9, 1.0], "interview_completion (real)")
hist(df.sig_offer_acceptance[df.sig_offer_acceptance >= 0], [0, .2, .4, .6, .8, .9, 1.0], "offer_acceptance (real)")
hist(df.sig_assessment_mean.dropna(), [0, 20, 40, 50, 60, 70, 80, 90, 100], "assessment_mean (real)")

print("\n" + "=" * 80)
print("S2. CORRELATIONS + LATENT FACTOR (PC1)")
print("=" * 80)
corr = df[NUMSIGS].corr()
print("pairs with |r| > 0.10:")
pairs = []
for i in range(len(NUMSIGS)):
    for j in range(i + 1, len(NUMSIGS)):
        r = corr.iloc[i, j]
        if abs(r) > 0.10:
            pairs.append((abs(r), NUMSIGS[i], NUMSIGS[j], r))
for _, a, b, r in sorted(pairs, reverse=True):
    print(f"  {a:30s} x {b:30s} r={r:+.3f}")
print(f"  (total pairs over threshold: {len(pairs)} of {len(NUMSIGS)*(len(NUMSIGS)-1)//2})")
off = corr.values[np.triu_indices(len(NUMSIGS), 1)]
print(f"  off-diagonal |r|: mean={np.abs(off).mean():.4f}  max={np.abs(off).max():.4f}")

# engagement-cluster signals (drop salary/notice which are logistics, drop sentinels by masking)
eng = ["sig_profile_completeness", "sig_profile_views_30d", "sig_applications_30d",
       "sig_response_rate", "sig_avg_response_hours", "sig_connection_count",
       "sig_endorsements_received", "sig_search_appearance_30d",
       "sig_saved_by_recruiters_30d", "days_since_active"]
X = StandardScaler().fit_transform(df[eng])
pca = PCA(n_components=3).fit(X)
pc = pca.transform(X)
print(f"\nPCA on 10 engagement signals: explained var = {pca.explained_variance_ratio_.round(3)}")
print("PC1 loadings:")
for c, l in sorted(zip(eng, pca.components_[0]), key=lambda t: -abs(t[1])):
    print(f"  {c:30s} {l:+.3f}")
hist(pd.Series(pc[:, 0]), bins=12, label="PC1")
sub = pc[np.random.choice(N, 20000, replace=False), :1]
print("GMM BIC on PC1 (lower=better; if k>=2 wins by a lot -> discrete tiers):")
for k in range(1, 5):
    g = GaussianMixture(k, random_state=0, n_init=2).fit(sub)
    print(f"  k={k}  BIC={g.bic(sub):,.0f}")

print("\n" + "=" * 80)
print("S3. RECENCY (days_since_active)")
print("=" * 80)
dsa = df.days_since_active
print(dsa.describe().round(1).to_string())
hist(dsa, [-1, 7, 30, 90, 180, 270, 400])
print(f"  inactive >90d: {(dsa>90).mean()*100:.2f}%   >180d: {(dsa>180).mean()*100:.2f}%")
ghost = dsa > 180
print("ghost (>180d) vs active means:")
for c in ["sig_response_rate", "sig_applications_30d", "sig_profile_views_30d",
          "sig_open_to_work", "sig_profile_completeness", "sig_saved_by_recruiters_30d"]:
    print(f"  {c:30s} ghost={df.loc[ghost,c].astype(float).mean():8.3f}  active={df.loc[~ghost,c].astype(float).mean():8.3f}")
print(f"corr(days_since_active, sig_response_rate)    = {dsa.corr(df.sig_response_rate):+.3f}")
print(f"corr(days_since_active, sig_applications_30d) = {dsa.corr(df.sig_applications_30d):+.3f}")
print(f"corr(days_since_active, sig_profile_views_30d)= {dsa.corr(df.sig_profile_views_30d):+.3f}")
print(f"signup_after_active=True (impossible dates): {df.signup_after_active.sum()} ({df.signup_after_active.mean()*100:.2f}%)")

print("\n" + "=" * 80)
print("S4. ARE SIGNALS CONDITIONED ON PROFILE QUALITY? (leak test)")
print("=" * 80)
AI_TITLE = re.compile(
    r"machine learning|\bml\b|\bai\b|artificial intelligence|data scien|deep learning|"
    r"\bnlp\b|computer vision|\bllm\b|mlops|research scientist|applied scientist", re.I)
title_txt = (df.current_title.fillna("") + " | " + df.headline.fillna(""))
ai_title = title_txt.str.contains(AI_TITLE)
yoe_band = df.yoe.between(5, 9)
ai_fit = ai_title & yoe_band
print(f"AI-relevant title: {ai_title.sum()} ({ai_title.mean()*100:.2f}%); +5-9 yoe: {ai_fit.sum()} ({ai_fit.mean()*100:.2f}%)")

JD_KW = re.compile(
    r"embedding|vector (db|database|search)|retriev|faiss|pinecone|weaviate|milvus|qdrant|"
    r"\brag\b|semantic search|hybrid search|learning.to.rank|ndcg|\bmrr\b|recommend(er|ation)|"
    r"ranking|elasticsearch|opensearch|two.tower|fine.tun|lora|peft", re.I)
prof_txt = (df.headline.fillna("") + " " + df.summary.fillna("") + " " +
            df.skill_names.fillna("").astype(str) + " " + df.job_descriptions.fillna("").astype(str))
kw_hits = prof_txt.str.count(JD_KW)
jd_match = (kw_hits >= 3) & yoe_band
print(f"JD-keyword match (>=3 hits + 5-9 yoe): {jd_match.sum()} ({jd_match.mean()*100:.2f}%)")

ALL_NUM = NUMSIGS
for name, mask in [("AI title", ai_title), ("AI title + 5-9yoe", ai_fit), ("JD-match", jd_match)]:
    print(f"\n--- target: {name} (n_pos={mask.sum()}) ---")
    rows = []
    for c in ALL_NUM:
        v = df[c].fillna(df[c].median())
        auc = roc_auc_score(mask, v)
        rows.append((abs(auc - .5), c, auc, df.loc[mask, c].mean(), df.loc[~mask, c].mean()))
    rows.sort(reverse=True)
    print("  per-signal AUC (vs 0.5 = no info), top 8:")
    for _, c, auc, mp, mn in rows[:8]:
        print(f"    {c:30s} AUC={auc:.4f}  mean_pos={mp:10.2f}  mean_neg={mn:10.2f}")
    Xs = df[ALL_NUM].fillna(df[ALL_NUM].median())
    Xs = StandardScaler().fit_transform(Xs)
    Xb = np.hstack([Xs, df[BOOLSIGS].values.astype(float),
                    pd.get_dummies(df.sig_work_mode).values.astype(float)])
    lr = LogisticRegression(max_iter=2000)
    aucs = cross_val_score(lr, Xb, mask, cv=3, scoring="roc_auc")
    print(f"  LogReg(all 23 signals) 3-fold AUC = {aucs.mean():.4f} +/- {aucs.std():.4f}")

print("\n" + "=" * 80)
print("S5. GITHUB ACTIVITY vs TECHNICAL TITLES")
print("=" * 80)
gh = df.sig_github_activity
TECH = re.compile(r"engineer|developer|scientist|architect|devops|sre|programmer|\bml\b|\bai\b|data", re.I)
tech = title_txt.str.contains(TECH)
print(f"technical-title share: {tech.mean()*100:.2f}%")
print(f"gh==-1: overall {(gh==-1).mean()*100:.2f}%  tech {(gh[tech]==-1).mean()*100:.2f}%  non-tech {(gh[~tech]==-1).mean()*100:.2f}%")
print(f"gh>0 mean: tech {gh[tech & (gh>=0)].mean():.2f}  non-tech {gh[~tech & (gh>=0)].mean():.2f}")
print(f"gh>=50: tech {(gh[tech]>=50).mean()*100:.2f}%  non-tech {(gh[~tech]>=50).mean()*100:.2f}%")
print(f"AUC(gh predicts tech title) = {roc_auc_score(tech, gh):.4f}")
print(f"gh==-1 rate for AI-fit: {(gh[ai_fit]==-1).mean()*100:.2f}%  JD-match: {(gh[jd_match]==-1).mean()*100:.2f}%")

print("\n" + "=" * 80)
print("S6. AVAILABILITY COHORT (verified + open_to_work + recent)")
print("=" * 80)
v2 = df.sig_verified_email & df.sig_verified_phone
print(f"verified email&phone: {v2.mean()*100:.2f}%   +linkedin: {(v2 & df.sig_linkedin_connected).mean()*100:.2f}%")
ct = pd.crosstab([df.sig_verified_email, df.sig_verified_phone],
                 [df.sig_open_to_work, df.sig_linkedin_connected])
print(ct.to_string())
exp = (df.sig_verified_email.mean() * df.sig_verified_phone.mean() *
       df.sig_open_to_work.mean() * df.sig_linkedin_connected.mean())
joint = (v2 & df.sig_open_to_work & df.sig_linkedin_connected)
print(f"all-4 joint: observed {joint.mean()*100:.3f}%  vs independent-expected {exp*100:.3f}%")
avail = joint & (dsa <= 30) & (df.sig_response_rate >= .5)
print(f"'deliberately available' (all-4 + active<=30d + response>=50%): {avail.sum()} ({avail.mean()*100:.2f}%)")
print(f"  ...of which AI title: {ai_title[avail].mean()*100:.2f}% (base {ai_title.mean()*100:.2f}%)  JD-match: {jd_match[avail].mean()*100:.2f}% (base {jd_match.mean()*100:.2f}%)")

print("\n" + "=" * 80)
print("S7. NOTICE PERIOD + SALARY")
print("=" * 80)
print(df.sig_notice_period_days.value_counts().sort_index().to_string())
print(f"notice<30d: {(df.sig_notice_period_days<30).mean()*100:.2f}%  <=30: {(df.sig_notice_period_days<=30).mean()*100:.2f}%  >=60: {(df.sig_notice_period_days>=60).mean()*100:.2f}%")
print(f"corr(notice, yoe)={df.sig_notice_period_days.corr(df.yoe):+.3f}")
print(f"salary_min: median={df.sig_salary_min.median():,.0f}  salary_max median={df.sig_salary_max.median():,.0f}")
print(f"corr(salary_min, yoe)={df.sig_salary_min.corr(df.yoe):+.3f}  corr(salary_max,yoe)={df.sig_salary_max.corr(df.yoe):+.3f}")
print("salary_min mean by yoe bucket:")
print(df.groupby(pd.cut(df.yoe, [0, 3, 5, 9, 15, 50]))["sig_salary_min"].mean().round(0).to_string())
print("notice mean by yoe bucket:")
print(df.groupby(pd.cut(df.yoe, [0, 3, 5, 9, 15, 50]))["sig_notice_period_days"].mean().round(1).to_string())
print(f"AI-fit (title+5-9yoe) cohort n={ai_fit.sum()}: notice<=30d {((df.sig_notice_period_days<=30)&ai_fit).sum()}"
      f" ({(df.sig_notice_period_days[ai_fit]<=30).mean()*100:.1f}%);  <30d {((df.sig_notice_period_days<30)&ai_fit).sum()}")
print(f"JD-match cohort n={jd_match.sum()}: notice<=30d {((df.sig_notice_period_days<=30)&jd_match).sum()}"
      f" ({(df.sig_notice_period_days[jd_match]<=30).mean()*100:.1f}%)")
print(f"salary_max < salary_min count: {(df.sig_salary_max < df.sig_salary_min).sum()}")

print("\n" + "=" * 80)
print("S8. INTERVIEW COMPLETION / OFFER ACCEPTANCE")
print("=" * 80)
ic, oa = df.sig_interview_completion, df.sig_offer_acceptance
print(f"interview_completion real values: {(ic>=0).mean()*100:.2f}%   offer_acceptance: {(oa>=0).mean()*100:.2f}%")
print(f"both real: {((ic>=0)&(oa>=0)).mean()*100:.2f}%   both -1: {((ic<0)&(oa<0)).mean()*100:.2f}%")
print(f"real ic: mean={ic[ic>=0].mean():.3f} med={ic[ic>=0].median():.3f}   real oa: mean={oa[oa>=0].mean():.3f} med={oa[oa>=0].median():.3f}")
print(f"corr(ic, oa | both real) = {ic[(ic>=0)&(oa>=0)].corr(oa[(ic>=0)&(oa>=0)]):+.3f}")
print(f"oa==-1 but ic>=0: {((oa<0)&(ic>=0)).mean()*100:.2f}%   ic==-1 but oa>=0: {((ic<0)&(oa>=0)).mean()*100:.2f}%")

print("\n" + "=" * 80)
print("S9. BEHAVIORAL TWINS PROBE (same profile, different signals?)")
print("=" * 80)
key = (df.headline.fillna("") + "||" + df.current_title.fillna("") + "||" +
       df.current_company.fillna("") + "||" + df.yoe.astype(str) + "||" + df.skill_names.fillna("").astype(str))
vc = key.value_counts()
dupes = vc[vc > 1]
print(f"duplicate profile keys (headline+title+company+yoe+skills): {len(dupes)} groups, {dupes.sum()} candidates, max group={dupes.max() if len(dupes) else 0}")
if len(dupes):
    gk = dupes.index[0]
    grp = df[key == gk]
    show = ["candidate_id", "sig_response_rate", "days_since_active", "sig_open_to_work",
            "sig_notice_period_days", "sig_profile_completeness", "sig_applications_30d"]
    print("example twin group:")
    print(grp[show].to_string(index=False))
    # within-group signal spread vs global
    samp = dupes.index[:200]
    sub2 = df[key.isin(samp)].copy()
    sub2["k"] = key[key.isin(samp)]
    wg = sub2.groupby("k")[["sig_response_rate", "days_since_active"]].std().mean()
    print(f"mean within-twin-group std: response_rate={wg.iloc[0]:.3f} (global std {df.sig_response_rate.std():.3f}), "
          f"days_since_active={wg.iloc[1]:.1f} (global std {dsa.std():.1f})")

print("\n" + "=" * 80)
print("S10. ANOMALIES IN SIGNALS (honeypot-relevant?)")
print("=" * 80)
inv = df.sig_salary_max < df.sig_salary_min
print(f"salary_max < salary_min: {inv.sum()} ({inv.mean()*100:.2f}%)  corr(min,max)={df.sig_salary_min.corr(df.sig_salary_max):+.3f}")
print(f"  inverted rows: AI-title share {ai_title[inv].mean()*100:.2f}% (base {ai_title.mean()*100:.2f}%)  JD-match {jd_match[inv].mean()*100:.2f}% (base {jd_match.mean()*100:.2f}%)")
saa = df.signup_after_active
print(f"signup_after_active: {saa.sum()} ({saa.mean()*100:.2f}%)")
# verify directly from the date strings
su = pd.to_datetime(df.sig_signup_date)
la = pd.to_datetime(df.sig_last_active_date)
print(f"  verified from raw dates signup>last_active: {(su>la).sum()};  max gap days: {(su-la).dt.days.max()}")
print(f"  signup_after_active rows: AI-title {ai_title[saa].mean()*100:.2f}%  JD-match {jd_match[saa].mean()*100:.2f}%  date_duration_mismatch {df.date_duration_mismatch[saa].mean()*100:.2f}% (base {df.date_duration_mismatch.mean()*100:.2f}%)")
rare_notice = df.sig_notice_period_days.isin([0, 15, 45])
print(f"rare notice values (0/15/45d): {rare_notice.sum()} candidates; AI-title {ai_title[rare_notice].mean()*100:.1f}%  JD-match {jd_match[rare_notice].mean()*100:.1f}%")
print(f"  sample ids: {df.loc[rare_notice, 'candidate_id'].head(8).tolist()}")
print(f"yoe>15 cohort: n={(df.yoe>15).sum()}  salary_min mean={df.loc[df.yoe>15,'sig_salary_min'].mean():.1f}  notice mean={df.loc[df.yoe>15,'sig_notice_period_days'].mean():.1f}")
print("yoe deciles vs salary_min / notice:")
print(df.groupby(pd.qcut(df.yoe, 8, duplicates='drop'), observed=True).agg(
    n=("yoe", "size"), sal_min=("sig_salary_min", "median"), notice=("sig_notice_period_days", "mean")).to_string())

print("\n" + "=" * 80)
print("S11. ASSESSMENTS JSON: topics as relevance leak?")
print("=" * 80)
topics = {}
ai_topic_score = np.full(N, np.nan)
AI_TOPICS = {"NLP", "Fine-tuning LLMs", "Vector Databases", "Information Retrieval", "Recommender Systems",
             "Machine Learning", "Deep Learning", "Python", "Search & Ranking", "Embeddings", "MLOps"}
has_ai_topic = np.zeros(N, bool)
for i, js in enumerate(df.sig_assessments_json.values):
    o = json.loads(js)
    for k, v in o.items():
        topics.setdefault(k, []).append(v)
        if k in AI_TOPICS:
            has_ai_topic[i] = True
            ai_topic_score[i] = max(ai_topic_score[i], v) if not np.isnan(ai_topic_score[i]) else v
print(f"distinct topics: {len(topics)}")
for k in sorted(topics, key=lambda k: -len(topics[k])):
    print(f"  {k:30s} n={len(topics[k]):6d}  mean={np.mean(topics[k]):.1f}")
print(f"has >=1 AI-relevant topic: {has_ai_topic.mean()*100:.2f}%  | among JD-match {has_ai_topic[jd_match.values].mean()*100:.2f}%  among rest {has_ai_topic[~jd_match.values].mean()*100:.2f}%")
print(f"AUC(has_ai_topic -> JD-match) = {roc_auc_score(jd_match, has_ai_topic.astype(float)):.4f}")
m = ~np.isnan(ai_topic_score)
print(f"AI-topic max score: mean(JD-match)={np.nanmean(ai_topic_score[jd_match.values & m]):.1f}  mean(rest)={np.nanmean(ai_topic_score[~jd_match.values & m]):.1f}")

print("\n" + "=" * 80)
print("S12. CONTROLLED LEAK TEST: within 5-9 yoe only")
print("=" * 80)
band = yoe_band.values
sub_df, sub_y = df[band], jd_match[band]
print(f"5-9 yoe pool: {band.sum()}, JD-match within: {sub_y.sum()} ({sub_y.mean()*100:.2f}%)")
rows = []
for c in ALL_NUM:
    v = sub_df[c].fillna(sub_df[c].median())
    auc = roc_auc_score(sub_y, v)
    rows.append((abs(auc - .5), c, auc))
rows.sort(reverse=True)
for _, c, auc in rows[:8]:
    print(f"  {c:30s} AUC={auc:.4f}")
Xs = sub_df[ALL_NUM].fillna(sub_df[ALL_NUM].median())
Xs = StandardScaler().fit_transform(Xs)
Xb = np.hstack([Xs, sub_df[BOOLSIGS].values.astype(float), pd.get_dummies(sub_df.sig_work_mode).values.astype(float)])
aucs = cross_val_score(LogisticRegression(max_iter=2000), Xb, sub_y, cv=3, scoring="roc_auc")
print(f"LogReg(all signals | 5-9 yoe) 3-fold AUC = {aucs.mean():.4f} +/- {aucs.std():.4f}")
# and with has_ai_topic added
Xb2 = np.hstack([Xb, has_ai_topic[band].reshape(-1, 1).astype(float)])
aucs2 = cross_val_score(LogisticRegression(max_iter=2000), Xb2, sub_y, cv=3, scoring="roc_auc")
print(f"  + has_ai_assessment_topic feature: AUC = {aucs2.mean():.4f}")

print("\n" + "=" * 80)
print("S13. HONEYPOT-FLAG CANDIDATES vs SIGNALS (tier-0 envelope test)")
print("=" * 80)
hp_flags = {
    "date_duration_mismatch": df.date_duration_mismatch.astype(bool),
    "n_expert_zero_months>0": df.n_expert_zero_months > 0,
    "overlap_months>24": df.overlap_months > 24,
    "signup_after_active": df.signup_after_active.astype(bool),
}
check = ["sig_response_rate", "sig_profile_completeness", "sig_search_appearance_30d",
         "sig_saved_by_recruiters_30d", "days_since_active", "sig_assessment_mean"]
for fname, fmask in hp_flags.items():
    print(f"{fname}: n={fmask.sum()}")
    line = "   "
    for c in check:
        line += f" {c.replace('sig_','')}={df.loc[fmask,c].mean():.2f}/{df.loc[~fmask,c].mean():.2f}"
    print(line + "   (flagged/clean)")

print("\n" + "=" * 80)
print("S14. PC1 TAIL COMPOSITION (high-engagement cohort)")
print("=" * 80)
pc1 = pc[:, 0]
tail = pc1 > 3
print(f"PC1>3: n={tail.sum()}  AI-title {ai_title.values[tail].mean()*100:.1f}%  JD-match {jd_match.values[tail].mean()*100:.1f}%  (bases {ai_title.mean()*100:.1f}%/{jd_match.mean()*100:.1f}%)")
top1k = np.argsort(-pc1)[:1000]
print(f"top-1000 by PC1: AI-title {ai_title.values[top1k].mean()*100:.1f}%  JD-match {jd_match.values[top1k].mean()*100:.1f}%  yoe-5-9 {yoe_band.values[top1k].mean()*100:.1f}% (base {yoe_band.mean()*100:.1f}%)")

print("\n" + "=" * 80)
print("S9b. FUZZY TWINS PROBE (looser keys)")
print("=" * 80)
for kname, k in [("title+company+yoe", df.current_title.fillna("") + "|" + df.current_company.fillna("") + "|" + df.yoe.astype(str)),
                 ("job_companies seq", df.job_companies.fillna("").astype(str)),
                 ("skills set", df.skill_names.fillna("").astype(str)),
                 ("summary text", df.summary.fillna(""))]:
    vc2 = k[k.str.len() > 8].value_counts()
    d2 = vc2[vc2 > 1]
    print(f"  key={kname:20s} dupe groups={len(d2)}  candidates in dupes={d2.sum()}  max group={d2.max() if len(d2) else 0}")

print("\n" + "=" * 80)
print("S15. RARE NOTICE VALUES DEEP DIVE (possible tier marker)")
print("=" * 80)
for v in [0, 15, 30, 45, 60]:
    m = df.sig_notice_period_days == v
    print(f"notice={v:3d}: n={m.sum():6d}  AI-title={ai_title[m].mean()*100:5.1f}%  JD-match={jd_match[m].mean()*100:5.1f}%  "
          f"yoe5-9={yoe_band[m].mean()*100:5.1f}%  resp_rate={df.loc[m,'sig_response_rate'].mean():.3f}  dsa={df.loc[m,'days_since_active'].mean():.0f}")
rn = df[rare_notice]
print(f"\nrare-notice (0/15/45) n={len(rn)}; titles:")
print(rn.current_title.value_counts().head(10).to_string())
print(f"mean yoe={rn.yoe.mean():.1f}; open_to_work={rn.sig_open_to_work.mean()*100:.0f}%; resp_rate={rn.sig_response_rate.mean():.3f} (base {df.sig_response_rate.mean():.3f})")
print(f"dsa={rn.days_since_active.mean():.0f} (base {dsa.mean():.0f}); search_app={rn.sig_search_appearance_30d.mean():.0f} (base {df.sig_search_appearance_30d.mean():.0f})")
print(f"assessment n>0: {(rn.sig_n_assessments>0).mean()*100:.0f}% (base {(df.sig_n_assessments>0).mean()*100:.0f}%)")
print(f"first 20 ids: {rn.candidate_id.head(20).tolist()}")

print("\n" + "=" * 80)
print("S16. THE 100 SKILLS-SET TWIN PAIRS (behavioral twins trap?)")
print("=" * 80)
sk = df.skill_names.fillna("").astype(str)
vc3 = sk[sk.str.len() > 8].value_counts()
twin_keys = vc3[vc3 > 1].index
tw = df[sk.isin(twin_keys)].copy()
tw["tkey"] = sk[sk.isin(twin_keys)]
print(f"twin candidates: {len(tw)} in {len(twin_keys)} pairs")
same_cols = ["name", "headline", "current_title", "current_company", "yoe", "summary", "job_companies", "location"]
for c in same_cols:
    eq = tw.groupby("tkey")[c].nunique()
    print(f"  identical within pair - {c:16s}: {(eq==1).mean()*100:.0f}% of pairs")
sig_cols = ["sig_response_rate", "days_since_active", "sig_open_to_work", "sig_notice_period_days",
            "sig_applications_30d", "sig_profile_completeness", "sig_search_appearance_30d", "sig_interview_completion"]
print("within-pair signal differences (mean |delta|) vs expected for random pairs (=std*1.128):")
for c in sig_cols:
    dlt = tw.groupby("tkey")[c].agg(lambda s: abs(float(s.iloc[0]) - float(s.iloc[-1]))).mean()
    print(f"  {c:28s} pair|delta|={dlt:8.2f}   random-pair expected={df[c].astype(float).std()*1.128:8.2f}")
print(f"twin AI-title share: {ai_title[sk.isin(twin_keys)].mean()*100:.1f}%  JD-match: {jd_match[sk.isin(twin_keys)].mean()*100:.1f}%")
ex = tw[tw.tkey == twin_keys[0]]
print("example pair:")
print(ex[["candidate_id", "name", "current_title", "yoe", "sig_response_rate", "days_since_active",
          "sig_open_to_work", "sig_notice_period_days", "sig_search_appearance_30d"]].to_string(index=False))
print(f"twin ids (first 12): {tw.candidate_id.head(12).tolist()}")

print("\n" + "=" * 80)
print("S17. ASSESSMENT TOPIC POOLS: generic vs JD-stack envelope")
print("=" * 80)
tcounts = {k: len(v) for k, v in topics.items()}
jd_pool = {k for k, n in tcounts.items() if n < 600}
gen_pool = {k for k, n in tcounts.items() if n >= 600}
print(f"JD-stack pool: {len(jd_pool)} topics, total assessments={sum(tcounts[k] for k in jd_pool)}")
print(f"generic pool:  {len(gen_pool)} topics, total assessments={sum(tcounts[k] for k in gen_pool)}")
has_jd_topic = np.zeros(N, bool)
has_gen_topic = np.zeros(N, bool)
for i, js in enumerate(df.sig_assessments_json.values):
    o = json.loads(js)
    for k in o:
        if k in jd_pool:
            has_jd_topic[i] = True
        else:
            has_gen_topic[i] = True
print(f"candidates w/ >=1 JD-stack topic: {has_jd_topic.sum()} ({has_jd_topic.mean()*100:.2f}%)")
print(f"  of those: AI-title={ai_title.values[has_jd_topic].mean()*100:.1f}%  JD-match={jd_match.values[has_jd_topic].mean()*100:.1f}%  yoe5-9={yoe_band.values[has_jd_topic].mean()*100:.1f}%")
print(f"candidates w/ >=1 generic topic: {has_gen_topic.sum()} ({has_gen_topic.mean()*100:.2f}%)")
print(f"  of those: AI-title={ai_title.values[has_gen_topic].mean()*100:.1f}%  JD-match={jd_match.values[has_gen_topic].mean()*100:.1f}%")
both = has_jd_topic & has_gen_topic
print(f"both pools: {both.sum()}   JD-pool x rare-notice overlap: {(has_jd_topic & rare_notice.values).sum()}")
print(f"P(JD-match | jd_topic)={jd_match.values[has_jd_topic].mean()*100:.1f}% vs P(JD-match)={jd_match.mean()*100:.1f}%  lift={jd_match.values[has_jd_topic].mean()/jd_match.mean():.1f}x")

print("\n" + "=" * 80)
print("S18. GHOST ARCHETYPE + SMALL ANOMALY COHORT IDS")
print("=" * 80)
ghost2 = (df.sig_response_rate <= 0.07) & (dsa > 150)
print(f"ghosts (response<=7% & inactive>150d): {ghost2.sum()} ({ghost2.mean()*100:.2f}%)  AI-title {ai_title[ghost2].mean()*100:.1f}%  JD-match {jd_match[ghost2].mean()*100:.1f}%")
g_jd = ghost2 & jd_match
print(f"  ghost x JD-match (perfect-on-paper-but-gone trap): {g_jd.sum()}  ids: {df.loc[g_jd,'candidate_id'].head(10).tolist()}")
print(f"n_expert_zero_months>0 ids (README honeypot example, n={(df.n_expert_zero_months>0).sum()}): {df.loc[df.n_expert_zero_months>0,'candidate_id'].tolist()}")
print(f"date_duration_mismatch ids (n={df.date_duration_mismatch.sum()}): {df.loc[df.date_duration_mismatch.astype(bool),'candidate_id'].head(15).tolist()}")
print(f"yoe>15 ids (n={(df.yoe>15).sum()}): {df.loc[df.yoe>15,'candidate_id'].tolist()}")
print(f"  yoe>15 titles: {df.loc[df.yoe>15,'current_title'].tolist()}")

print("\n" + "=" * 80)
print("S19. TWIN PAIR VERIFICATION (designed or chance?)")
print("=" * 80)
g = tw.groupby("tkey")
n_opp = (g["sig_open_to_work"].nunique() == 2).sum()
print(f"pairs where open_to_work differs: {n_opp}/100 (chance p=0.5 each -> binomial)")
n_resp_big = (g["sig_response_rate"].agg(lambda s: abs(s.iloc[0] - s.iloc[-1])) > 0.4).sum()
n_dsa_big = (g["days_since_active"].agg(lambda s: abs(s.iloc[0] - s.iloc[-1])) > 100).sum()
print(f"pairs with |resp_rate delta|>0.4: {n_resp_big}/100;  |dsa delta|>100: {n_dsa_big}/100")
# is the 'available' twin the AI-relevant one?
tw["ai"] = ai_title[sk.isin(twin_keys)].values
open_ai = tw.loc[tw.sig_open_to_work, "ai"].mean()
closed_ai = tw.loc[~tw.sig_open_to_work, "ai"].mean()
print(f"AI-title share: open_to_work twin={open_ai*100:.0f}%  closed twin={closed_ai*100:.0f}%")
avail_t = tw.loc[tw.sig_open_to_work]
ghost_t = tw.loc[~tw.sig_open_to_work]
print(f"open twin:  resp={avail_t.sig_response_rate.mean():.2f} dsa={avail_t.days_since_active.mean():.0f} yoe={avail_t.yoe.mean():.1f}")
print(f"closed twin: resp={ghost_t.sig_response_rate.mean():.2f} dsa={ghost_t.days_since_active.mean():.0f} yoe={ghost_t.yoe.mean():.1f}")
shared = ["job_descriptions", "edu_institutions", "cert_names", "job_titles"]
for c in shared:
    eq = g[c].nunique()
    print(f"  identical within pair - {c:16s}: {(eq==1).mean()*100:.0f}% of pairs")

print("\n" + "=" * 80)
print("S20. SIGNAL-ONLY RANKING: who lands in a top-100 by signals alone?")
print("=" * 80)
z = lambda s: (s - s.mean()) / s.std()
avail_score = (z(df.sig_response_rate) - z(dsa) + z(df.sig_search_appearance_30d)
               + z(df.sig_saved_by_recruiters_30d) + z(df.sig_profile_completeness)
               + df.sig_open_to_work.astype(float) - (df.sig_notice_period_days >= 60).astype(float)
               + (df.sig_n_assessments > 0).astype(float))
for k in [100, 500, 1000]:
    top = np.argsort(-avail_score.values)[:k]
    print(f"top-{k:4d} by availability-composite: AI-title={ai_title.values[top].mean()*100:5.1f}%  "
          f"JD-match={jd_match.values[top].mean()*100:5.1f}%  rare-notice={rare_notice.values[top].mean()*100:4.1f}%  "
          f"jd_topic={has_jd_topic[top].mean()*100:4.1f}%")
top100 = np.argsort(-pc1)[:100]
print(f"top-100 by PC1(engagement): AI-title={ai_title.values[top100].mean()*100:.1f}%  JD-match={jd_match.values[top100].mean()*100:.1f}%")
print(f"rare-notice candidates in availability top-1000: {rare_notice.values[np.argsort(-avail_score.values)[:1000]].sum()} of 166")

print("\n" + "=" * 80)
print("S21. FINAL CHECKS: clipping masses, cohort overlaps, disqualifier mix in rare-notice")
print("=" * 80)
for c in ["sig_avg_response_hours", "sig_response_rate", "sig_interview_completion", "days_since_active"]:
    mx, mn = df[c].max(), df[c].min()
    print(f"{c}: mass at max({mx})={(df[c]==mx).sum()}  at min({mn})={(df[c]==mn).sum()}")
open_twin_ids = set(tw.loc[tw.sig_open_to_work, "candidate_id"])
rare_ids = set(rn.candidate_id)
print(f"open-twin x rare-notice overlap: {len(open_twin_ids & rare_ids)} of 100 open twins")
risk = rn.current_title.str.contains(r"junior|computer vision|research", case=False)
print(f"rare-notice members with risky titles (junior/CV/research): {risk.sum()} of {len(rn)}")
print(f"rare-notice yoe<5: {(rn.yoe<5).sum()};  yoe in [5,9]: {rn.yoe.between(5,9).sum()}")
# closed-twin profile: do they hold full AI skill lists with non-AI jobs (keyword stuffer)?
ghost_t2 = tw.loc[~tw.sig_open_to_work]
print(f"closed twins: n_skills mean={ghost_t2.n_skills.mean():.1f}; sample titles: {ghost_t2.current_title.value_counts().head(6).to_dict()}")
print(f"closed twin ids (first 10): {ghost_t2.candidate_id.head(10).tolist()}")
print(f"open twin sample skills (first pair): {ex.iloc[0]['candidate_id']} -> {str(df.loc[df.candidate_id==ex.iloc[0]['candidate_id'],'skill_names'].iloc[0])[:200]}")

print("\nDONE")
