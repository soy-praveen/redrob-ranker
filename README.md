# Redrob Ranker — Intelligent Candidate Discovery & Ranking Challenge

Ranks 100,000 candidate profiles against the Senior AI Engineer JD the way a
recruiter would: by reading career histories for real evidence, rejecting
profiles that only *look* relevant, and weighing whether a candidate is
actually reachable and hireable.

## Reproduce the submission

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

- **Runtime:** ~15 seconds for 100k candidates on a laptop CPU (limit: 5 min)
- **Memory:** < 4 GB (limit: 16 GB)
- **Network:** none — no API calls of any kind during ranking
- **Deterministic:** identical output on every run
- Accepts `candidates.jsonl` or `candidates.jsonl.gz`

Validate with the organizer's checker:

```bash
python scripts/validate_submission.py submission.csv
```

## How it works

A recruiter doesn't keyword-match — they read the career history, check the
claims line up, and ask "can we actually hire this person?" The ranker
implements exactly that, in four explicit, auditable scoring layers:

```
final = Content x Fit x Availability + Prior
```

**Stage 1 — Hard exclusions.** Profiles a recruiter would bin immediately:

- *Impossible profiles* (the dataset's planted honeypots): expert-proficiency
  skills with 0 months of use, job durations contradicting their own start/end
  dates, claimed experience exceeding the entire career span, summaries that
  state a different experience number than the profile field, certifications
  dated in the future. Five independent profile-internal consistency checks;
  any hit excludes the candidate.
- *Keyword stuffers*: non-technical titles (HR Manager, Sales Executive, ...)
  carrying 7-13 AI skills with no machine-learning evidence anywhere in their
  career history. Skills lists are cheap to fake; career history is not.
- *Cloned skill lists*: profiles whose exact skill set duplicates another
  candidate's — the dataset copies genuine AI engineers' skill lists onto
  unrelated profiles. The career-incoherent member of each pair is dropped.

**Stage 2 — Content score (what they've actually done).** Career-history
text is matched against a graded evidence rubric (production
retrieval/ranking/recsys systems score highest, applied-ML production work
mid, ML-adjacent infra low) that was hand-graded against the JD's
must-haves and re-graded by a blind 3-judge panel. Crucially this reads only
the *career history*, never the skills list or self-summary — so candidates
who describe ranking systems in plain language score highly even with zero
buzzwords, and buzzword-stuffed profiles with empty careers score zero.
Skill claims contribute only when corroborated (long durations, real
endorsements, matching assessment scores).

**Stage 3 — Fit multiplier (the JD's explicit rules).** Experience band
(6-8 yrs ideal, 5-9 in-band, tapering outside), location ladder
(Pune/Noida > Hyderabad/Mumbai/Delhi NCR > rest of India > abroad-willing-to-
relocate > abroad), consulting-only career demotion, job-hopper demotion —
each rule traceable to a sentence in the JD.

**Stage 4 — Availability multiplier (can we actually hire them?).** The JD
is explicit that a perfect-on-paper candidate who ignores recruiters is not
a real candidate. Graded on recruiter response rate, recency of activity,
notice period, and open-to-work status, with a hard demotion for ghosts
(response < 20% and inactive > 100 days). Multiplicative — chosen over an
additive term after simulation showed additive weighting lets
behaviorally-dead profiles crowd out hireable ones.

**Stage 5 — Engagement prior (small, additive).** Recruiter-side demand
signals (search appearances, recruiter saves, profile views), completed
retrieval-stack skill assessments, and public GitHub activity.

**Reasoning column.** Generated from each candidate's own profile facts:
their strongest career evidence, concrete signal values, and honest concerns
(long notice, inactivity, out-of-band experience...). No claim appears that
is not in the profile.

## Repository layout

```
rank.py                 # CLI entry point (the reproduce command)
src/features.py         # JSONL -> 55-feature table (one streaming pass)
src/scoring.py          # exclusions + C/F/A/P scoring + assembly checks
src/reasoning.py        # per-candidate recruiter-style reasoning
config/params.json      # the full scoring rubric: paragraph grades, weights
scripts/                # data prep + the organizer's validator
docs/                   # JD, submission spec, schema (converted to markdown)
eda/                    # the analysis that produced every parameter above
sandbox/                # Streamlit demo app (hosted sandbox)
```

## Why not embeddings / an LLM?

We tried the obvious architecture first on paper and rejected it from the
data up. The dataset (like real recruiting corpora) is adversarial:
profiles stuffed with retrieval keywords, near-identical texts differing
only in behavioral signals, and impossible profiles designed to punish
systems that score surface similarity. Embedding similarity rewards exactly
those traps, misses plain-language descriptions of ranking work, and costs
1000x more compute. Reading structured evidence from career history with a
graded rubric is deterministic, auditable, runs in 15 seconds on CPU — and
is what a great recruiter actually does. The full analysis (two rounds,
multiple independent reviews, an adversarial audit of the final output) is
in `eda/`.

## Verification before submission

`rank.py` asserts at assembly time: exactly 100 unique candidates, zero
honeypot-marker hits, zero stuffers, monotone non-increasing scores with
the validator's tie-break rule, and prints the India share and evidence-
grade composition of the final list.
