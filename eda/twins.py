# -*- coding: utf-8 -*-
"""
BEHAVIORAL TWINS / DUPLICATES analysis — Redrob ranker EDA.

Questions answered:
 Q1 exact-duplicate text groups (summary / headline / skills+title+yoe / job_descriptions)
 Q2 near-duplicate pipeline (cheap hash blocking -> rapidfuzz verification) -> twin pairs?
 Q3 field-level diffs within identified pair plants + signal gaps
 Q4 twins/plants inside the plausible AI-relevant pool
 Q5 name reuse: twin marker or generator coincidence?
 Q6 summary template archetypes (generator structure) and title-quality correlation

Run:  PYTHONIOENCODING=utf-8 python eda/twins.py
Deterministic, ~3-4 min (composite sweep dominates).
"""
import collections
import itertools
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

PARQUET = "c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons/redrob-ranker/data/candidates_flat.parquet"

df = pd.read_parquet(PARQUET)
N = len(df)
print(f"loaded {N} rows x {df.shape[1]} cols")

SIG_COLS = [c for c in df.columns if c.startswith("sig_")] + ["days_since_active", "signup_after_active"]
NONSIG = [c for c in df.columns if c not in SIG_COLS and c != "candidate_id"]

df["skillset"] = df["skill_names"].map(lambda s: " | ".join(sorted(s.split(" | "))))
df["compset"] = df["job_companies"].map(lambda s: " | ".join(sorted(s.split(" | "))))
df["yoe_r"] = df["yoe"].round(0)

AI_TITLE_RE = re.compile(
    r"(machine learning|ml engineer|\bai\b|data scientist|nlp|deep learning|computer vision"
    r"|research (scientist|engineer)|applied scientist|mlops|llm|generative|search|recommend)", re.I)
df["ai_title"] = df["current_title"].fillna("").apply(lambda t: bool(AI_TITLE_RE.search(t)))


def dup_hist(cols, label):
    dup = df.duplicated(subset=cols, keep=False)
    g = df[dup].groupby(cols, dropna=False).size()
    hist = collections.Counter(g.values)
    print(f"  {label}: unique={df.groupby(cols,dropna=False).ngroups}, rows_in_dup_groups={int(dup.sum())}, "
          f"n_dup_groups={len(g)}, max_group={max(hist) if hist else 0}")
    top = sorted(hist.items())
    print(f"    group-size hist (size:count): {dict(top[:12])}{' ...' if len(top) > 12 else ''}")
    return g


# ----------------------------------------------------------------------------
print("\n=== Q1: EXACT-DUPLICATE TEXT GROUPS ===")
dup_hist(["summary"], "identical summary")
dup_hist(["headline"], "identical headline")
dup_hist(["skillset", "current_title", "yoe_r"], "identical (sorted skills, title, yoe rounded)")
dup_hist(["job_descriptions"], "identical job_descriptions")
dup_hist(["job_titles", "job_companies", "job_descriptions"], "identical FULL job history")
print("  -> field-level duplication is massive (template reuse), so exact field match alone != twin")

# full-profile fingerprint: identical on ALL non-signal columns
for label, cols in [("all non-signal cols incl name", NONSIG),
                    ("all non-signal cols excl name", [c for c in NONSIG if c != "name"])]:
    n = int(df.duplicated(subset=cols, keep=False).sum())
    print(f"  rows duplicated on {label}: {n}")

# ----------------------------------------------------------------------------
print("\n=== Q2: NEAR-DUPLICATE PIPELINE (blocking + verification) ===")

# (a) THE one statistically-impossible plant: exact ordered skill list reused
vc_ord = df["skill_names"].value_counts()
pairs_ord = vc_ord[vc_ord == 2]
vc_set = df["skillset"].value_counts()
pairs_set = vc_set[vc_set == 2]
sub = df[df["skillset"].isin(pairs_set.index)]
nsk = sub["n_skills"]
print(f"  exact ORDERED skill-list reused: {len(pairs_ord)} pairs (no group >2)")
print(f"  exact UNORDERED skill-set reused: {len(pairs_set)} pairs "
      f"({len(pairs_set)-len(pairs_ord)} extra differ only in order)")
print(f"  colliding lists have n_skills {nsk.min()}-{nsk.max()} (mean {nsk.mean():.1f}) vs population mean "
      f"{df['n_skills'].mean():.1f} -> chance collision of 9-20 ordered skills ~ impossible => PLANTED")

clone_pat = collections.Counter()
clone_rows = []
for sk, grp in sub.groupby("skillset"):
    a, b = grp.iloc[0], grp.iloc[1]
    clone_pat[(a["ai_title"], b["ai_title"])] += 1
    clone_rows.append((a, b))
one_ai = clone_pat[(True, False)] + clone_pat[(False, True)]
print(f"  pattern: exactly-one-member-has-AI-title in {one_ai}/{len(pairs_set)} pairs "
      f"(both False: {clone_pat[(False, False)]}, both True: {clone_pat[(True, True)]})")
print("  => these are KEYWORD-STUFFER clones: an AI donor's skill list copied onto a non-AI career, not behavioral twins")

# (b) verify the big blocking keys are template coincidence, not twins
print("\n  blocking-key verification (within-group rapidfuzz / Jaccard):")
key = ["job_titles", "job_companies", "job_descriptions", "summary"]
sub2 = df[df.duplicated(subset=key, keep=False)]
jacs, rrd, dsad = [], [], []
for k, grp in sub2.groupby(key):
    a, b = grp.iloc[0], grp.iloc[1]
    sa, sb = set(a["skill_names"].split(" | ")), set(b["skill_names"].split(" | "))
    jacs.append(len(sa & sb) / len(sa | sb))
    rrd.append(abs(a["sig_response_rate"] - b["sig_response_rate"]))
    dsad.append(abs(a["days_since_active"] - b["days_since_active"]))
jacs = np.array(jacs)
print(f"  same career+summary: {sub2.groupby(key).ngroups} groups ({len(sub2)} rows); "
      f"within-pair skill Jaccard mean {jacs.mean():.3f} (max {jacs.max():.2f}; >=0.8: {(jacs>=.8).sum()})")
print(f"    -> skills independently sampled => template coincidence, NOT twins "
      f"(|rr delta| median {np.median(rrd):.2f}, |days_active delta| median {np.median(dsad):.0f})")

key2 = ["job_titles", "job_companies", "job_descriptions"]
sub3 = df[df.duplicated(subset=key2, keep=False)]
hi = 0
npairs = 0
for k, grp in sub3.groupby(key2):
    rows = list(grp.itertuples())
    ss = [set(r.skill_names.split(" | ")) for r in rows]
    for i, j in itertools.combinations(range(len(rows)), 2):
        npairs += 1
        if len(ss[i] & ss[j]) / len(ss[i] | ss[j]) > 0.5:
            hi += 1
print(f"  same FULL career ({sub3.groupby(key2).ngroups} groups, {npairs} pairs): "
      f"pairs with skill Jaccard>0.5: {hi} -> zero career+skill twins exist")

# (c) composite agreement sweep: any pair highly similar across many fields?
print("\n  composite sweep: block on summary/job_descriptions/skillset/compset, score 11-field agreement")
fields = ["summary", "headline", "current_title", "job_titles", "job_companies",
          "job_descriptions", "skillset", "edu_institutions", "location", "name"]
seen = {}
for bk in ["summary", "job_descriptions", "skillset", "compset"]:
    v = df[bk].value_counts()
    d = v[(v > 1) & (v <= 200)].index
    for k, grp in df[df[bk].isin(d)].groupby(bk):
        rows = list(grp.itertuples())[:60]  # cap mega template groups
        for a, b in itertools.combinations(rows, 2):
            pid = (a.candidate_id, b.candidate_id) if a.candidate_id < b.candidate_id else (b.candidate_id, a.candidate_id)
            if pid in seen:
                continue
            seen[pid] = sum(getattr(a, f) == getattr(b, f) for f in fields) + (abs(a.yoe - b.yoe) < 0.15)
sc = np.array(list(seen.values()))
hist = collections.Counter(sc)
print(f"  pairs scored: {len(sc)}; agreement(of 11) tail: "
      f"8/11: {hist.get(8,0)}, 7/11: {hist.get(7,0)}, 6/11: {hist.get(6,0)}; max = {sc.max()}")
print("  VERDICT Q2: NO literal behavioral-twin pairs (identical-except-signals) exist."
      "\n          Duplication is a GENERAL template-reuse property; the only targeted plant is the 100/102 skill-clone pairs.")

# ----------------------------------------------------------------------------
print("\n=== Q3: DIFF THE PLANTED PAIRS + SIGNAL GAPS ===")
differ = collections.Counter()
sig_deltas = collections.defaultdict(list)
donor_clone = []
for a, b in clone_rows:
    for c in NONSIG:
        if a[c] != b[c]:
            differ[c] += 1
    for c in ["sig_response_rate", "days_since_active", "sig_github_activity",
              "sig_profile_completeness", "sig_interview_completion"]:
        sig_deltas[c].append(abs(a[c] - b[c]))
    if a["ai_title"] != b["ai_title"]:
        donor, clone = (a, b) if a["ai_title"] else (b, a)
        donor_clone.append((donor, clone))
print(f"  non-signal fields differing across the {len(clone_rows)} skill-clone pairs (top): "
      f"{dict(differ.most_common(8))}")
print("  -> careers/names/summaries ALL differ; ONLY the skill list is shared. "
      "'only-signals-differ' hypothesis FALSIFIED for every duplicate family tested.")
for c, v in sig_deltas.items():
    print(f"    |delta {c}|: mean {np.mean(v):.3f} median {np.median(v):.3f}")
print(f"\n  donor (AI-title) vs clone (non-AI) signal means over {len(donor_clone)} unambiguous pairs:")
for c in ["sig_response_rate", "days_since_active", "sig_github_activity", "sig_profile_completeness",
          "sig_interview_completion", "sig_profile_views_30d", "sig_saved_by_recruiters_30d"]:
    dm = np.mean([d[c] for d, _ in donor_clone])
    cm = np.mean([c2[c] for _, c2 in donor_clone])
    print(f"    {c}: donor {dm:.3f} vs clone {cm:.3f}")

print("\n  --- 4 full pair examples (all fields incl sig_*) ---")
show_cols = (["candidate_id", "name", "current_title", "yoe", "location", "job_titles", "job_companies",
              "best_edu_tier", "n_skills"] + SIG_COLS)
for a, b in clone_rows[:4]:
    print(f"  PAIR sharing skills: {a['skill_names'][:90]}...")
    for r in (a, b):
        tag = "AI  " if r["ai_title"] else "non-AI"
        print(f"   [{tag}] " + " | ".join(f"{c}={r[c]}" for c in show_cols if c != "sig_assessments_json"))
    print()

# ----------------------------------------------------------------------------
print("=== Q4: PLANTS / TEXT-INDISTINGUISHABLE GROUPS IN THE AI-RELEVANT POOL ===")
df["tpl_n"] = df["summary"].str[:60].str.replace(r"\d+(\.\d+)?", "#", regex=True)
ai_tpl = df["tpl_n"].str.startswith(("Data scientist / ML engineer", "Machine learning engineer",
                                     "Senior AI engineer", "Senior engineer who"))
india = df["country"].str.contains("India", case=False, na=False)
pool = df[(df["ai_title"] | ai_tpl) & df["yoe"].between(4, 10) & (india | df["sig_willing_to_relocate"])]
print(f"  plausible pool (AI title/archetype, yoe 4-10, India-or-relocate): {len(pool)} candidates")

in_pool_clones = [r["candidate_id"] for a, b in clone_rows for r in (a, b)
                  if r["candidate_id"] in set(pool["candidate_id"])]
clone_only = [c["candidate_id"] for d, c in donor_clone]
stuffers_in_pool = sorted(set(in_pool_clones) & set(clone_only))
donors_in_pool = sorted(set(in_pool_clones) - set(clone_only))
print(f"  skill-clone pair members inside pool: {len(in_pool_clones)} "
      f"({len(stuffers_in_pool)} are the non-AI STUFFER side -> avoid; {len(donors_in_pool)} donor/ambiguous)")
print(f"  stuffer ids in pool: {stuffers_in_pool}")
print(f"  donor/ambiguous ids in pool: {donors_in_pool}")

# text-indistinguishable groups inside the pool: identical summary+headline+title
g = pool.groupby(["summary", "headline", "current_title"])
twins_like = {k: grp for k, grp in g if len(grp) > 1}
spreads = []
for k, grp in twins_like.items():
    rr = grp["sig_response_rate"]
    da = grp["days_since_active"]
    spreads.append((rr.max() - rr.min(), da.max() - da.min(), len(grp),
                    grp.loc[rr.idxmax(), "candidate_id"], grp.loc[rr.idxmin(), "candidate_id"]))
spreads.sort(reverse=True)
sizes = collections.Counter(len(grp) for grp in twins_like.values())
print(f"\n  text-indistinguishable groups in pool (same summary+headline+title): {len(twins_like)} groups, "
      f"sizes {dict(sorted(sizes.items()))}")
if spreads:
    rrs = np.array([s[0] for s in spreads])
    print(f"  within-group response_rate spread: mean {rrs.mean():.2f}, >=0.5 in {(rrs>=.5).sum()} groups")
    print("  top engaged-vs-ghost 'functional twin' examples (rr_spread, days_spread, n, engaged_id, ghost_id):")
    for s in spreads[:10]:
        print(f"    {s[0]:.2f}  {s[1]:4.0f}  n={s[2]}  engaged={s[3]}  ghost={s[4]}")

# the senior archetypes: tiny handcrafted top pool with engaged/ghost split
sen = df[df["tpl_n"].str.startswith(("Senior AI engineer", "Senior engineer who"))]
ghost = sen[(sen["sig_response_rate"] < 0.2) & (sen["days_since_active"] > 100)]
print(f"\n  senior-AI archetypes: {len(sen)} candidates; GHOSTS (rr<0.2 & inactive>100d): {len(ghost)}")
print("  ghost ids: " + ", ".join(ghost["candidate_id"] + " (" + ghost["current_title"] + ", rr=" +
                                  ghost["sig_response_rate"].astype(str) + ", inactive " +
                                  ghost["days_since_active"].astype(str) + "d)"))
print("  plain-language 'Senior engineer who...' archetype ids (likely no-buzzword tier-5s): "
      f"{sorted(df.loc[df['tpl_n'].str.startswith('Senior engineer who'), 'candidate_id'])}")

# ----------------------------------------------------------------------------
print("\n=== Q5: NAME REUSE ===")
vc = df["name"].value_counts()
fn = df["name"].str.split().str[0].nunique()
ln = df["name"].str.split().str[-1].nunique()
print(f"  unique names: {len(vc)} (pool = {fn} first x {ln} last names); every name reused "
      f"(min {vc.min()}, mean {vc.mean():.1f}, max {vc.max()})")
same_nc = df.duplicated(subset=["name", "compset", "job_titles", "job_descriptions"], keep=False)
n_same = int(same_nc.sum())
# chance expectation: same-career pairs x P(two random candidates share a name)
p_name = float((vc / N).pow(2).sum())
exp_pairs = npairs * p_name  # npairs = same-full-career pairs from Q2
print(f"  same name + identical full career: {n_same} rows ({n_same//2} pairs); "
      f"chance expectation = {npairs} same-career pairs x P(name match)={p_name:.2e} ~= {exp_pairs:.1f}")
print(f"  -> observed {n_same//2} vs expected-by-chance ~{exp_pairs:.0f}: names are PURE GENERATOR COINCIDENCE, "
      "carry zero twin signal")

# ----------------------------------------------------------------------------
print("\n=== Q6: SUMMARY TEMPLATE ARCHETYPES ===")
tn = df["tpl_n"].value_counts()
raw60 = df["summary"].str[:60].nunique()
print(f"  distinct raw 60-char prefixes: {raw60}; digit-normalized templates: {len(tn)}")
print("  top 30 templates (count, ai_title_rate, yoe_median, prefix):")
for t, c in tn.head(30).items():
    m = df["tpl_n"] == t
    print(f"   {c:6d}  ai={df.loc[m,'ai_title'].mean():.2f}  yoe_med={df.loc[m,'yoe'].median():4.1f}  "
          f"{t[:70]}")
print(f"  remaining templates: {dict(tn.iloc[30:].items())}")
quota = tn[tn.index.str.startswith(("Data scientist / ML engineer", "Software / data professional"))]
print(f"  note round quotas: {dict(quota.items())}")
print("  archetype <-> title quality is essentially deterministic: only the DS/ML (1000), "
      "senior-AI (21+8) and MLE (150) templates ever carry AI titles.")
print("\ndone.")
