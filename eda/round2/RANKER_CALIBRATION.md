# RANKER CALIBRATION — final parameter sheet for rank.py

Synthesized 2026-06-10 from round-2 analysts: grades-reconciliation, top-15 ordering, honeypot-final, tail-strategy. All numbers computed, not estimated. Reference date for every recency gate: **2026-06-10**.

Canonical inputs (all verified present):
- `eda/round2/paragraph_grades_consensus.json` — 44 pids, 3-judge median grades. **MUST be read with `encoding='utf-8'`** (cp1252 mojibake silently breaks the P06/P12/P22 em-dash fingerprints; P22 is the CV-borderline detector).
- `eda/round2/honeypot_final.csv` — 93 exclusion ids (candidate_id, markers, ai_titled).
- `eda/stuffer_ids.csv` — 5,517 stuffer ids.
- `eda/round2/top15_ordering.md`, `eda/round2/tail_strategy.md` — head + tail rationale.
- Data: `data/candidates_flat.parquet` (100k × 67); raw JSONL for assembly-time marker re-derivation.

---

## 1. Final paragraph grade table (pid → grade, match on 60-char prefix of job_description paragraphs)

Consensus = 3-judge median, zero overrides. Matching: exact 60-char-prefix fingerprint against each candidate's job_descriptions; `best_grade` = max over a candidate's paragraphs. Strata over 100k (clean = after honeypot exclusion): grade-5 = 29 raw / **27 clean** (21 live + 6 ghosts); grade-4 = 139 / **129**; grade-3 = 963 / **961**; clean grade≥4 = **156**.

| pid | grade | prefix (60 chars) |
|---|---|---|
| P00 | 0 | `Enterprise sales of cloud software solutions into the mid-ma` |
| P01 | 0 | `Customer support team lead at a SaaS product. Managed a team` |
| P02 | 0 | `Marketing leadership role at a B2B SaaS company. Owned the d` |
| P03 | 0 | `Business analyst at a consulting firm, working primarily wit` |
| P04 | 0 | `Brand design and creative direction at a consumer-products c` |
| P05 | 0 | `Mechanical engineering design role at a hardware-product com` |
| P06 | 0 | `Senior accounting role at a mid-sized company — month-end cl` |
| P07 | 0 | `Content writing and SEO strategy for a tech-focused publicat` |
| P08 | 0 | `Operations management role at a logistics company. Owned dai` |
| P09 | 1 | `Cloud infrastructure and DevOps work at an enterprise SaaS c` |
| P10 | 1 | `Android mobile development using Java and (more recently) Ko` |
| P11 | 1 | `Frontend engineering at a media company. React, TypeScript, ` |
| P12 | 1 | `Java backend development at a large enterprise — Spring Boot` |
| P13 | 1 | `Full-stack web application development at a SaaS company. Bu` |
| P14 | 1 | `Test automation and QA engineering for a fintech product. Bu` |
| P15 | 2 | `Designed and maintained the analytical data warehouse on Sno` |
| P16 | 2 | `Built and maintained data pipelines on Apache Airflow proces` |
| P17 | 2 | `Backend + data hybrid role at a growth-stage startup. Built ` |
| P18 | 2 | `Implemented streaming data pipelines on Kafka and Spark Stre` |
| P19 | 2 | `Mixed data science and analytics-engineering role at a marke` |
| P20 | 2 | `Backend development with Python (FastAPI), PostgreSQL, and R` |
| P21 | 3 | `Contributed to ML feature engineering and model deployment f` |
| P22 | 3 | `Built recommendation-style features at a mid-stage startup —` |
| P23 | 2 | `Built computer vision models for our product's image moderat` |
| P24 | 3 | `Worked on time-series forecasting models for supply-chain de` |
| P25 | 3 | `Worked on customer-facing predictive modeling for an e-comme` |
| P26 | 3 | `Built NLP pipelines for sentiment analysis and document clas` |
| P27 | 4 | `Owned the ranking layer for an e-commerce search product, ev` |
| P28 | 4 | `Trained and shipped multiple ranking models for our product'` |
| P29 | 4 | `Developed a semantic search feature for an internal knowledg` |
| P30 | 3 | `Implemented a RAG-based customer support chatbot integrated ` |
| P31 | 4 | `Built a content recommendation system serving 10M+ users tha` |
| P32 | 3 | `Built and operated production ML pipelines using MLflow for ` |
| P33 | 5 | `Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA and` |
| P34 | 5 | `Built a RAG-based ranking pipeline serving 50M+ queries per ` |
| P35 | 5 | `Built and shipped a production recommendation system at a ma` |
| P36 | 5 | `Owned the end-to-end ranking pipeline at a recommendations-h` |
| P37 | 5 | `Owned the design and rollout of a large-scale semantic searc` |
| P38 | 5 | `Led the migration from keyword-based to embedding-based sear` |
| P39 | 5 | `Built systems that understand what users are looking for and` |
| P40 | 5 | `Shipped the personalization infrastructure: the system that ` |
| P41 | 5 | `Designed the ranking layer for the company's flagship produc` |
| P42 | 5 | `Owned the search and discovery experience end-to-end at a co` |
| P43 | 5 | `Led the engineering team building infrastructure to surface ` |

Key consensus moves vs round-1: P27/P28/P29/P31 demoted 5→4; P22 and P30 demoted 4→3 (JD anti-pattern / no-eval hedges — never re-admit to the ≥4 pool); P24 promoted 2→3; P23 stays 2 (CV disqualifier pattern). Downstream code must import THIS table, not `tier5-language.py::PARA_GRADES`.

---

## 2. Exclusion list policy (run BEFORE pool construction — 14 honeypots live inside the grade-3+ pools, 2 inside the elite 29)

**E1. Honeypots — hard-exclude all 93 ids in `eda/round2/honeypot_final.csv`.** Every id re-verified from raw JSONL; 5 marker families, each a planted island with a dead zone below it:
- A: any skill proficiency=expert AND duration_months=0
- B: any job |duration_months − months(start,end)| > 3
- C: yoe·12 > career_span_months + 24
- D: |summary-text yoe − yoe field| > 1.5 (both directions)
- E: any cert year > 2026 (only 2030 exists: "AWS Certified Machine Learning Specialty")

Never emit **CAND_0039754** or **CAND_0093547** (the 2 grade-5 elite-template baits). At assembly, re-derive all 5 markers from raw (per `honeypot_crosscheck.py::verify_raw`) — do not trust the CSV alone. Treat "~80" as resolved (≈ A+B+E primary-corruption = 79); do not hunt for a smaller set.

**E2. NOT honeypot rules (would delete plants):** pre-founding tenures (>50% of all Sarvam/Krutrim tenures violate; plant CAND_0006567 itself violates), skill-duration-vs-tech-age (7/8 plants trip it), salary inversion, signup_after_active, reversed degrees, saved>views.

**E3. Stuffers — exclude the 5,517** via summary substring `"online courses on RAG and vector databases"` (cross-check vs `stuffer_ids.csv`).

**E4. Skill-set clones (~102 unordered-duplicate pairs):** demote the career-incoherent (non-AI-titled) member to bottom; the donor side is legitimate.

**E5. Non-tech titles (12 personas, 68,821):** exclude wholesale from the top-100 pool (verified zero-cost: 0 built-ML evidence among them).

**E6. CV-titled (132):** out of top-100 entirely. The 34 admissible-with-penalty CV+P22 borderlines go only to the 100-200 reserve (see §4), title-penalized below same-evidence AI-titled grade-3s; CAND_0088385 penalized hardest (genuinely CV-dominant).

---

## 3. Score formula

**S = C × F × A + P**, applied WITHIN bands; band membership (grade strata + pinned head/ghosts) dominates (see §4). P is small and must never flip a grade band.

### C — content (0–112)
- Base: `C = 20 × best_grade` (100/80/60/40/20/0 — the grade gaps the tail simulation validated).
- +6 per additional distinct grade≥4 paragraph beyond the best (cap +12).
- +8 head-family bonus if T_ELITE or T_PLAIN summary template. **One feature, not two:** T_ELITE membership and HR-domain paragraphs (P33/P34/P38) are perfectly collinear (20/21 elites hold one; nobody else in 100k does) — do NOT add a separate HR-paragraph bonus on top, it would double-count the axis against T_PLAIN.
- +4 if any of the 14 paraphrased skill tokens (Search Infrastructure, Vector Representations, Ranking Systems, …) — T_PLAIN backup detector.
- +2 per JD-core skill with duration ≥18mo (cap +6).
- −15 if best paragraph is P23 (CV-only); −10 if best is P30 (RAG-wrapper-only); −2 per hedge phrase (cap −6).
- Within-grade adjustments are capped so they never exceed the 20-point grade gap.

### F — fit (multiplier, 0.30–1.16)
- Agreed yoe = median{yoe field, summary-text yoe, career_span/12} (never the field alone): 6–8 → ×1.05; 5–9 → ×1.00; 4–5 or 9–10 → ×0.90; else ×0.70.
- Location ladder: Pune/Noida ×1.10; NCR/Hyderabad/Mumbai ×1.05; other India (incl. Bangalore) ×1.00; abroad + willing_to_relocate ×0.85; abroad no-relocate ×0.60.
- Consulting-ONLY career at the six JD-named firms (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini — six-firm definition is canonical) ×0.50; mixed exposure: no penalty.
- Hopper: avg_tenure <20mo AND ≥2 ended short stints ×0.92 (mild, per JD "demote, don't kill").

### A — availability (MULTIPLICATIVE, per tail simulation; additive w≤0.2 was the single worst configuration tested)
`A = amin + (1.1 − amin) × A01` with **amin = 0.5** (flat-optimal anywhere in 0.40–0.75; cliff above 0.75 — never raise amin past 0.75).
- `A01 = 0.45·clip(resp/0.8,0,1) + 0.30·clip(1 − days_active/120,0,1) + 0.15·notice_norm + 0.10·open_to_work` where notice_norm: ≤30 → 1.0; 45–60 → 0.7; 90 → 0.4; ≥120 → 0.1.
- **Hard ghost floor:** resp <0.2 AND days_active >100 ⇒ A = amin (0.5), regardless of A01.
- If an additive form is ever reintroduced it must carry an explicit ghost-demotion ≥ one full grade gap (~25 points on the C scale).

### P — prior (additive, z-scored, capped |P| ≤ 8)
- Engagement composite (search_appearance_30d, saved_by_recruiters_30d, views_30d, completeness): up to +4.
- JD-stack assessment presence/score (34 topics): up to +2.
- github_activity ≥50: +1 (−1 = missing, never zero).
- Notice ∈ {0,15,45} envelope-leak flag: +1.5 (apply after fit gates; 166 ids, 100% core-AI/CV).
- Salary archetype (salary_min ≥ ~19 LPA): +0.5. Never penalize high asks.
- Do NOT use signal clustering to promote G4s to tier-5 (G4 signals are unimodal; the only bimodality is the 6 ghosts). Signals = ghost rule + smooth within-grade ordering only.

---

## 4. Assembly: top-15 hand-ordering + ranks 16–100

### Ranks 1–15 (pinned, exactly this order)
1. CAND_0046525 — ELITE, Pune, 6.1y, P34+P38, 0.88/18d
2. CAND_0061257 — PLAIN, Noida, 8.0y, 0.87/55d
3. CAND_0018499 — ELITE, Noida, 7.2y, P34×2+P33, notice-15 leak
4. CAND_0006567 — PLAIN, Noida, 7.9y
5. CAND_0068351 — PLAIN, Delhi NCR, 6.4y, notice-0 leak
6. CAND_0077337 — ELITE, Kochi(reloc), 7.0y, 0.95/15d, P38×2+P37
7. CAND_0081846 — ELITE, Jaipur(reloc), 6.7y, P34
8. CAND_0011687 — ELITE, Indore, 7.8y, P33, notice-15 leak
9. CAND_0030468 — PLAIN, Indore(reloc), 5.4y, notice-45 leak
10. CAND_0093193 — PLAIN, Bangalore, 7.9y, notice-45 leak
11. CAND_0005260 — ELITE, Chennai(reloc), 5.2y, P33+P34
12. CAND_0002025 — ELITE, Trivandrum, 5.9y, P33
13. CAND_0037980 — PLAIN, Kolkata(reloc), 9.0y, notice-0 leak
14. CAND_0086022 — ELITE, Kolkata(reloc), 5.3y, P34+P38, notice-0 leak
15. CAND_0080766 — PLAIN, Coimbatore(reloc), 8.8y, notice-0 leak

Top-10 = 5 ELITE + 5 PLAIN interleave (family-risk hedge: if either family is secretly tier-4, the other still holds alternating top slots). All 15 pass every raw-derived exclusion check (0 honeypot markers, 0 ghosts, 0 stuffers, 0 clones, 0 six-firm exposure).

Only contestable swap inside the top-8: rank 3 (CAND_0018499, resp 0.61, 3×HR-paragraph) vs rank 6 (CAND_0077337, resp 0.95). **Decision: keep 0018499 at 3** — Noida + notice-15 leak + triple HR-domain career outweighs the signal gap; swap only if a final sensitivity run weights envelopes over HR-paragraph density.

### Ranks 16–21 (pinned: remaining 6 live head members)
16. CAND_0005538 (PLAIN, planted job-hopper — demoted, not deleted; still tier-5 text)
17. CAND_0071974 · 18. CAND_0046064 · 19. CAND_0008425 · 20. CAND_0088025 · 21. CAND_0055905 (London no-reloc — kept last in head)

### Ranks 22–94: live grade-4 only (tail policy (c), the robust winner: mean 0.9412 / worst 0.9325 / regret 0.0137)
- Fill with the top 73 of the 129 clean live G4s ordered by S = C×F×A + P.
- The P27/P28/P29/P31 holders ARE this pool — sitting immediately behind the head hedges the 5-vs-4 grading boundary (judge 1 still said 5), per the grades analyst.
- Grade-3 is NEVER needed inside the top-100 (156 clean grade≥4 > 100 slots). No CV borderline in the top-100.
- Within-band tie-breaks: location ladder, then resp desc, days_active asc, notice ladder (with {0,15,45} leak bonus), hopper demotion below clean-tenure peers.

### Ranks 95–100: the 6 elite ghosts (pinned block)
CAND_0033861, CAND_0094759, CAND_0060072, CAND_0007411, CAND_0092278, CAND_0041611 (ordered by C×F with A floored; 0041611 last — Austin USA). Placement ~90–100 caps worst-case loss at −0.008 under ghost-tier-2 while keeping content-dominant upside and insuring against G4 over-grading. **Never above rank 60; never in 11–30** (would cost −0.044 at 11–16 under tier-2 vs +0.015 upside).

### Reserve 100–200 (only if the format scores deeper; MAP insurance)
Strong-G3 (336 P22/P30 holders) by C×A → other G3 → the 34 admissible CV+P22 borderlines (title-penalized one within-band step below same-evidence AI-titled G3s; order from tail analyst: CAND_0069638, CAND_0046924, CAND_0049217, CAND_0048469, CAND_0060340, CAND_0025663, CAND_0077296, CAND_0003506, CAND_0054703, CAND_0053934, CAND_0069953, CAND_0088385 last) → grade-2.

### Founding-suspect insurance (cheap DQ hedge, adopted)
Cap the 14-id founding-suspect watchlist (CAND_0011687, CAND_0017960, CAND_0088385, CAND_0011432, CAND_0005509, CAND_0008049, CAND_0024878, CAND_0077296, CAND_0085851, CAND_0051615, CAND_0069248, CAND_0015528, CAND_0018013, CAND_0060835) at **≤8 inside the top-100** (demote lowest-scored surplus to 101+, replace with next G4). CAND_0006567 (rank 4 plant) is exempt and must NOT be demoted.

---

## 5. Assembly-time assertions (run on the FINAL 100 before EVERY submission)

1. **Honeypots = 0:** re-derive all 5 markers (A–E) from raw JSONL on each of the 100 → 0 hits; intersect with the 93-id CSV → empty; explicitly assert CAND_0039754 and CAND_0093547 absent.
2. **Honeypot headroom:** even counting every founding-suspect in-list as hypothetically dirty, total ≤8 < 10 (the 10% DQ bar).
3. **Twins:** no skill-set-clone pair (unordered hash) has both members in the 100; for any text-identical (summary+headline+title) pair both present, the higher-signal member ranks higher.
4. **Stuffers/personas:** 0 stuffer-boilerplate summaries; 0 non-tech titles; 0 CV-titled.
5. **India share ≥ 90/100** (India or willing_to_relocate).
6. **Band monotonicity:** ranks 1–21 = the 21 live head (grade-5); 22–94 all best_grade ≥4 and sorted desc by S; 95–100 = exactly the 6 ghost ids; no resp<0.2 ∧ act>100d candidate anywhere above rank 94.
7. **Pinned-head integrity:** ranks 1–15 match §4 exactly; 100 unique ids total.
8. **Encoding/fingerprints:** consensus JSON read as UTF-8; all 44 prefixes match ≥1 candidate pool-wide (P22 must hit 369, P30 must hit 60 — mojibake canary).
9. **Date sensitivity:** all recency math assumes 2026-06-10; if rerun later, recompute days_since_active before gating.

## 6. Conflicts decided (tradeoffs)

| conflict | decision | why |
|---|---|---|
| Grades analyst: backfill ranks 51–100 from grade-3 vs tail analyst: grade≥4 covers all 100 | **Tail analyst** (no grade-3 in top-100) | Grades analyst wrote before the pool-level honeypot correction; 156 clean G4+ > 100; sim shows G3-fill dominated in all 6 GT cells |
| Top-15 analyst: ghosts "ranks ~50–100" vs tail analyst: "~90–100, never above 60" | **95–100 pinned block** | Quantified asymmetry: −0.044 at 11–16 / −0.026 at 30–60 under ghost-tier-2 vs +0.015 max upside; 95–100 caps loss at 0.008 |
| Honeypot analyst: founding-suspects out of ranks 1–10 vs top-15 analyst: CAND_0011687 at rank 8 | **Keep 0011687 at 8** + adopt the ≤8 cap | Founding violations proven generator noise 4 ways (plant CAND_0006567 itself violates at rank 4 — demoting the family is incoherent); 0011687 is raw-verified zero-marker; rank 8 already IS the de-risked placement; one id = 1% of the 10% DQ bar |
| Availability multiplicative vs additive | **Multiplicative, amin=0.5, hard ghost floor** | Additive w≤0.2 leaves ghost elites above live G4s (min crashes to 0.91); multiplicative flat-optimal 0.40–0.75 |
| T_ELITE vs T_PLAIN at the head | **Interleave 5/5 on logistics×signals** | Honeypots were planted inside T_ELITE ⇒ elite template is the tier-5 reference; per-candidate flaws, not family, drive ordering; interleave bounds family-level surprise |
