# ADVERSARIAL REVIEW — Final Verdict & Prioritized Fix List

Synthesized 2026-06-10 from four independent adversarial audits (HONEYPOT, REASONING,
RANK-QUALITY, REPRODUCIBILITY), each run against the RAW `candidates.jsonl` with no trust
in pipeline outputs. One inter-audit contradiction was found and resolved by independent
re-verification (section 3).

## 1. Overall Verdict: **PASS WITH FIXES**

| Audit | Verdict |
|---|---|
| Honeypot / DQ risk | PASS |
| Reasoning quality (Stage 4) | PASS_WITH_FIXES |
| Rank quality | PASS |
| Reproducibility (Stage 3) | PASS_WITH_FIXES |

**No disqualification risk.** 0/100 honeypot markers (A–E re-derived from raw by a fresh
parser, `eda/round3/audit_honeypots_fresh.py`); zero overlap with the 93-id
`eda/round2/honeypot_final.csv`; no clones; founding-suspect watchlist members in top-100 = 4
(cap is 8, DQ bar is 10). **Stage 3 is solid:** clean-room run (only rank.py + src/ + config)
reproduces `submission_draft.csv` byte-identically (sha256 `9ba8a6a1...a197`) across 3 runs
incl. `PYTHONHASHSEED` stress, in 11–15 s wall / 0.47–0.55 GB RSS vs the 5 min / 16 GB budget;
stdlib+pandas+numpy only; no network, no hardcoded ids. **No fabricated facts:** all 100
evidence clauses, titles, companies, yoe, response-%, notice-day and github claims verified
verbatim against raw.

The residual exposure is **Stage-4 manual reasoning audit** (five generation bugs that read
as hallucination/sloppiness) and **administrative/repo completeness** (placeholder metadata,
untracked deliverables). All are fixable in src/reasoning.py + repo hygiene without touching
the ranking itself (except the optional rank-70 swap).

> NOTE: any reasoning fix changes `submission_draft.csv` bytes. After fixes: rerun rank.py,
> rerun `scripts/validate_submission.py`, redo the 3-run determinism check, re-record sha256.

## 2. Prioritized Fix List (de-duplicated)

### CRITICAL — must fix before submission

**C1. Fill submission_metadata.yaml placeholders.**
`submission_metadata.yaml` lines 4–20 still contain FILL_ME/TODO for team_name,
primary_contact.name, phone, github_repo, sandbox_link. The file states Stage 3 verifies
against it — placeholders risk administrative rejection.
*File:* `submission_metadata.yaml`. *(Source: REPRODUCIBILITY)*

**C2. Commit untracked deliverables.**
README.md, requirements.txt, submission_metadata.yaml, and sandbox/ are all untracked
(`??` in git status). Pushed as-is, Stage 3/4 reviewers get a repo with no README, no
reproduce instructions, no requirements, and a dangling sandbox_link.
*Files:* git add/commit `README.md`, `requirements.txt`, `submission_metadata.yaml`,
`sandbox/`. *(Source: REPRODUCIBILITY)*

**C3. Misattributed evidence — 11 reasoning rows.**
Frame `"{Title} at {CurrentCompany} ({yoe}) — {evidence}"` attaches a PAST employer's work
to the current one. Worst: rank 44 CAND_0010257 reads "Senior Data Scientist at Google
(6.5 yrs) — shipped discovery-feed ranking models (XGBoost/LightGBM)…" but that paragraph is
from his PharmEasy stint; the actual Google job is MLflow/Kubeflow + churn. A Stage-4 auditor
reading only the current job description will call this a hallucination. Affected ranks:
33, 35, 44, 58, 76, 87, 91, 92, 93, 95, 99.
*Fix:* in `src/reasoning.py`, carry the source company alongside the evidence tag (extend
`_evidence` to return `(tag_text, company)`) and name it, or use neutral phrasing
("earlier shipped…", "previously built…"). *(Source: REASONING)*

**C4. Self-contradictory YoE — 6 rows.**
Lead clause uses `row['yoe']` (line 83) while `_concerns` uses `agreed_yoe`
(lines 45–48), producing two different numbers in one sentence. Rank 54 CAND_0099806
"(4.6 yrs)… at 4.5 yrs"; also ranks 63 (4.7/4.6), 83 (4.6/4.5), 88 (4.1/4.0), 89 (4.5/4.4),
99 (4.9/4.8). Instantly visible to a manual reviewer.
*Fix:* `src/reasoning.py` — use one value (recommend `agreed_yoe`) for both the lead
`{yoe}` and the concern string. *(Source: REASONING)*

**C5. Broken grammar in tail band — 4 rows.**
"Ranked low on slightly below the 5-9 yr band at 4.5 yrs" (ranks 83, 88, 89, 99): the
`"Ranked low on {joined}"` template (line 125) does not compose with the yoe-band concern
string (line 46). Reads as a generation bug.
*Fix:* `src/reasoning.py` — make every `_concerns` entry a noun phrase (e.g. "a sub-band
tenure of 4.5 yrs") or special-case the tail template. *(Source: REASONING)*

**C6. Rank 70 CAND_0055905 — abroad + not-willing-to-relocate, unacknowledged.**
The only non-India candidate (London, UK; willing_to_relocate=False; JD: no visa
sponsorship) has purely positive reasoning ("Responds to 87%…; active within the last
3 weeks"). Root cause: `_concerns` line 43 fires only on `location_tier == 0.5`, but
`location_tier()` (src/features.py lines 131–135) returns **0.0** for abroad+no-reloc — the
worst location case is silently dropped.
*Fix (both):* (a) add a tier-0.0 branch in `src/reasoning.py` _concerns: "based in
{country}, not open to relocation"; (b) **recommended** — swap CAND_0055905 out of the
top-100 entirely for the next clean live India G4 (boundary gap is only ~5 final-score
points per RANK-QUALITY; removes the issue and a JD-misfit slot at once).
*Files:* `src/reasoning.py` line 43; `src/scoring.py` (location-tier-0 penalty) if swapping.
*(Sources: REASONING critical + RANK-QUALITY minor — merged)*

### IMPORTANT — fix unless time-critical

**I1. Weeks-active recency overclaim — 18 rows (count resolved, see §3).**
Line 65 uses `round(dsa/7)`, so "active within the last N weeks" is literally false when
`dsa > N*7`. Verified 18/100 rows overstate by 1–3 days (e.g. r2 CAND_0006567 38d→"5 weeks";
r8 CAND_0011687 31d→"4 weeks"; r11 CAND_0046064 45d→"6 weeks"; full list in §3).
*Fix:* `src/reasoning.py` line 65 — `math.ceil(dsa/7)` or phrase "~N weeks ago".
*(Sources: REASONING + RANK-QUALITY)*

**I2. Cross-platform byte-identity of the CSV.**
`rank.py` line 67 `to_csv` uses the OS default lineterminator → CRLF on Windows; a Linux
Stage-3 reproduction yields LF and a byte-diff "fails" on every line despite identical
content. *Fix:* add `lineterminator='\n'` and regenerate the draft.
*File:* `rank.py` line 67. *(Source: REPRODUCIBILITY)*

**I3. Pin requirements exactly; drop stray pyarrow.**
Loose pins (`pandas>=2.0`, `numpy>=1.24`) and `pyarrow>=14.0` which the deliverable never
imports (only scripts/flatten.py needs it). *Fix:* `pandas==2.3.3`, `numpy==2.3.4`, note
python 3.13.2; remove pyarrow or comment it as optional-data-prep-only.
*File:* `requirements.txt`. *(Source: REPRODUCIBILITY)*

**I4. open_to_work=False is never surfaced (22 rows).**
JD stresses "Active on Redrob… so we can actually talk to them", yet otw=False appears in
22 top-100 rows (e.g. ranks 6, 16, 19, 26, 41, 58) with zero mentions; rank 16
CAND_0080766 even says "immediately available" (notice=0 real, but otw=False + 62d
inactive). *Fix:* add an otw=False entry to `_concerns` (at minimum for ranks > 35).
*File:* `src/reasoning.py`. *(Source: REASONING)*

**I5. Template visibility — 4 stock evidence clauses cover 79/100 rows verbatim.**
Three verbatim-identical evidence pairs landed inside the 10 mandated audit samples
(ranks 7&13, 22&71, 86&99). Partly defensible (the dataset itself duplicates
job-description paragraphs across candidates), but the skeleton is audit-visible.
*Fix:* add 2–3 hash-picked paraphrases per evidence tag (extend `config/params.json`
evidence_tags to lists; select via `_pick`).
*Files:* `config/params.json`, `src/reasoning.py`. *(Source: REASONING)*

**I6. Regenerate + revalidate after fixes.**
Rerun `rank.py`, `scripts/validate_submission.py`, and the 3-run determinism/sha check;
spot-reread the 11 C3 rows and the 4 C5 rows. *(Process item)*

### NICE — polish if time permits

**N1.** Lower the staleness-mention threshold from 90d to ~60d for tail-band rows: ~28 rows
sit silent in the 60–90d window (ranks 31, 41, 86, 99 at dsa 85–86d).
*File:* `src/reasoning.py` line 38. *(REASONING)*

**N2.** Rank 41 CAND_0087630: jointly weak availability (resp 0.45, otw=False, 86d
inactive, github 23) — the JD's "perfect-on-paper but unreachable" anti-profile. Add a
concern clause or demote slightly. *(REASONING)*

**N3.** Rank 23 CAND_0000031: job-hopper (3 stints <20 mo, avg 17.5 mo) at rank 23 with
only a x0.88 demotion. Defensible (grade-4 x4 strong paragraphs, resp 0.91, 17d active,
Hyderabad) but pre-write the rationale for Stage 4. *(RANK-QUALITY)*

**N4.** Rank 58 CAND_0044855 "Scored 70 avg on retrieval-stack skill assessments" is
correct only under the jd-stack subset (69.7→70); full-set avg is 68.9. List the skills in
the clause to preempt a reviewer recomputing the wrong average. *(REASONING)*

**N5.** Document the two deviations from `eda/round2/RANKER_CALIBRATION.md` before Stage 4:
(a) the 6 elite ghosts fully excluded instead of the pinned 95–100 insurance block
(forfeits bounded ~+0.015 MAP upside if tiers ignore liveness; own sim bounds delta ~0.01);
(b) pinned 1–15 hand-order replaced by pure formula order (rank 1 = CAND_0018499 resp 0.61,
not CAND_0046525). Both defensible, neither documented. *(RANK-QUALITY)*

**N6.** Remove or gitignore tracked pickles `eda/round2/_df_cache.pkl`, `_fp_cache.pkl`
(repo bloat / sloppy at code review). *(REPRODUCIBILITY)*

**N7.** README's provenance narrative points at eda/, but `eda/round3/`, `eda/review/`,
`eda/_audit_sample10.txt`, `eda/_audit_top100_raw.json` are untracked — commit them if the
story is meant to be inspectable. *(REPRODUCIBILITY)*

**N8.** `rank.py` lines 45–46 stuff shared dicts into DataFrame columns to reach
`generate_reasoning` — works and is deterministic, but pass them as function args.
*(REPRODUCIBILITY)*

**N9.** `submission_draft.csv` is .gitignored — confirm portal-upload-only is intentional;
some Stage-3 reviewers expect a committed file to diff against. *(REPRODUCIBILITY)*

**N10.** No action needed, recorded for completeness: the only belt-and-suspenders
honeypot swaps available are the 4 signup_date>last_active_date ids — CAND_0015528 (r42),
CAND_0061339 (r64), CAND_0064270 (r66), CAND_0064904 (r99). All verified as generator noise
(7,496/100k pool-wide) with zero co-occurring anomalies. Likewise verified-noise, not
markers: the 2018 LangChain cert at r17 (45/100 such certs pool-wide predate 2022; the real
cert marker uses year=2030 only) and the ~95mo skill-duration overhangs at r64/r66/r88
(generator hard-caps overhang at span+60; 0/100k above it). *(HONEYPOT)*

## 3. Contradiction Resolution (independently re-verified from raw)

**REASONING said 20/100 weeks-overclaim rows; RANK-QUALITY said 18/100.**
Re-derived from raw `candidates.jsonl` (`redrob_signals.last_active_date` vs
REFERENCE_DATE 2026-06-10) against the literal claims in `submission_draft.csv`:
26 rows carry an "active within the last N weeks" claim; exactly **18** have
`dsa > N*7`. **RANK-QUALITY's count is correct.** REASONING's list erroneously included
rank 56 (and its total of 20 is unsupported). Verified overclaim set
(rank, id, dsa, claim, overstatement):

```
r2  CAND_0006567 38d "5 weeks" +3d   r23 CAND_0000031 17d "2 weeks" +3d
r5  CAND_0077337 15d "2 weeks" +1d   r28 CAND_0093912 30d "4 weeks" +2d
r8  CAND_0011687 31d "4 weeks" +3d   r34 CAND_0050876 15d "2 weeks" +1d
r9  CAND_0030468 38d "5 weeks" +3d   r37 CAND_0036184 15d "2 weeks" +1d
r10 CAND_0081846 38d "5 weeks" +3d   r38 CAND_0018549 43d "6 weeks" +1d
r11 CAND_0046064 45d "6 weeks" +3d   r47 CAND_0076163 31d "4 weeks" +3d
r12 CAND_0002025 15d "2 weeks" +1d   r59 CAND_0017960 44d "6 weeks" +2d
r18 CAND_0005538 24d "3 weeks" +3d   r70 CAND_0055905 24d "3 weeks" +3d
r19 CAND_0005260 31d "4 weeks" +3d   r22 CAND_0050454 44d "6 weeks" +2d
```

The fix (I1) is identical either way. No other inter-audit contradictions: the rank-70
severity difference (REASONING critical vs RANK-QUALITY minor) is a judgment split, merged
as C6 with the swap recommended; honeypot-noise items (signup dates, LangChain cert, skill
overhang) were raised only by HONEYPOT and self-resolved with population-level evidence.

## 4. What Was Independently Confirmed Sound (do not re-litigate)

- 0/100 honeypot markers A–E from raw; 0 overlap with honeypot_final.csv; 0 skill-set clones.
- 21/21 clean elite-template holders included; all 8 elite/plain exclusions justified from
  raw (6 ghosts with resp 0.07–0.16 and 114–215d inactivity; 2 marker-bearing honeypots).
- Top-100 = 21 G5 + 79 G4, zero grade-3, zero stuffers/ghosts/consulting-only; 0
  strict-dominance violations in top-10; 0 adjacent inversions in ranks 11–50.
- All factual reasoning claims (evidence text, titles, companies, yoe, response-%,
  notice days, github) verified verbatim/exact against raw for all 100 rows.
- Byte-identical deterministic reproduction from a clean room in ~15 s / 0.5 GB; stdlib+
  pandas+numpy only; validator passes and matches the organizer's official copy byte-for-byte.
