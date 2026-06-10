# Tail Strategy & Borderline Analysis (Round 2)

Analyst: tail-strategy/borderline (MAP + NDCG@50 + ranks 11-100 composition).
Inputs: `data/candidates_flat.parquet`, consensus grades `eda/round2/paragraph_grades_consensus.json`, raw JSONL deep reads, simulator `eda/round2/tail_sim.py` (rerunnable; grid dump `_task2_grid.csv`). Reference date 2026-06-10. All numbers below are computed, not estimated.

## 0. Pool corrections that change the tail math

Consensus-grade fingerprinting over all 100k (0 unmatched paragraphs after fixing a UTF-8 mojibake in the consensus JSON — round-1's prefix file must be read with `encoding='utf-8'`):

| pool | raw | honeypots inside | clean |
|---|---|---|---|
| best-grade 5 (T_ELITE+T_PLAIN) | 29 | 2 (CAND_0039754, CAND_0093547) | **27** (21 live + 6 ghosts) |
| best-grade 4 | 139 | 10 | **129** (0 ghosts, 0 soft-dead) |
| best-grade 3 | 963 | 2 | **961** (0 hard ghosts, ~60 soft-dead) |
| grade >= 4 (round-1's "168") | 168 | **12** | **156** |

- **All 14 DQ-critical AI-bait honeypots sit inside the grade-3+ content pools** (2 in grade-5, 10 in grade-4, 2 in grade-3). The synthesis claim "zero overlap with the elite reads" is wrong at pool level — honeypot exclusion must run BEFORE tail assembly, not just as a post-check.
- 156 clean grade>=4 candidates > 100 slots: **the entire top-100 can be filled from grade>=4; grade-3 is never needed inside the top-100.**
- Tier-conditioned signal envelopes (exact, clean pools): live-elite resp ∈ [0.55, 0.95], act ≤ 74d; G4 resp ∈ [0.41, 0.94], act ≤ 89d; G3 resp ∈ [0.25, 0.95], act ≤ 114d; the 6 elite ghosts resp ∈ [0.07, 0.16], act 114-215d. Response-rate FLOOR rises with content grade — the generator conditions signals on tier, and the ghosts are a deliberately planted exception inside the elite tier.

## 1. CV-titled borderlines (the "45")

Exactly **45 of 132** Computer Vision Engineer-titled candidates hold P22 ("recommendation-style features ... collaborative filtering (matrix factorization) + gradient-boosted re-ranking, production") — none hold anything grade-4+. Only 43/132 CV-titled even hold the P23 CV paragraph (13 of the 45 do); 14/45 also hold P26 (DistilBERT NLP pipelines). Zero are honeypots, zero are ghosts.

Deep reads of 15 raw JSONL records (strongest by content+signals; full prints cached): the "CV Engineer" title is essentially a title-assignment artifact — summaries are the applied-ML-junior template ("strongest at the modeling and analysis side"), and career evidence is generic applied-ML (P21/P22/P24/P25/P26). **The JD disqualifier ("CV/speech/robotics-only WITHOUT NLP/IR exposure") does not literally fire for most**: P22 is production recsys/CF (IR-core domain) and P26 is real NLP. But none show any must-have (no embeddings-in-prod, no vector-DB ops, no ranking-eval frameworks), and P22 itself carries the "lighter weight than ranking systems at FAANG" hedge.

Verdicts (15 read):

| candidate | verdict | note |
|---|---|---|
| CAND_0069638 | admissible-w-penalty (strongest) | double P22 (63 months prod recsys, Swiggy+Zomato) + P26; Chennai, notice 30, resp 0.64, reloc |
| CAND_0046924 | admissible-w-penalty | P22 current + P22 prior + P21; **Noida**; resp 0.61, act 30d; notice 120 |
| CAND_0049217 | admissible-w-penalty | P22 current, resp 0.87, act 24d, Gurgaon; PhD-ML but production career (not pure-research) |
| CAND_0048469 | admissible-w-penalty | P22 current 36m + P26 + P21; Delhi |
| CAND_0060340 | admissible-w-penalty | P22 current + P26 + P21; notice 45 (envelope leak marker), act 16d |
| CAND_0025663 | admissible-w-penalty | P26 current + P22 + P25; notice 120 |
| CAND_0077296 | admissible-w-penalty | P22 old (Tech Mahindra); gh 76.9; notice 45 |
| CAND_0003506 | admissible-w-penalty (weak) | P22 5y old, recent work is fraud/forecasting |
| CAND_0054703 | admissible-w-penalty (weak) | has P23 but only 6m early-career; P22 37m at InMobi; resp 0.41 |
| CAND_0053934 | admissible-w-penalty (weak) | resp 0.32; P22 old; current Infosys |
| CAND_0069953 | admissible-w-penalty (weak) | P22 at Tech Mahindra; two consulting stints (not consulting-only) |
| CAND_0088385 | marginal — penalize hardest | the one genuinely CV-dominant read: 49m CV job (P23) at Glance dwarfs a 13m P22 |
| CAND_0001302 | exclude | Sydney, not willing to relocate |
| CAND_0058882 | exclude | yoe 3.6, P22 only 9m |
| CAND_0095207 | exclude (yoe) | yoe 4.0; otherwise good signals (resp 0.72, gh 83.8, Weaviate 81.5 assessment) |

Aggregate over all 45 (rules validated on the reads): 4 abroad-no-relocate, 8 yoe<4.5, 0 ghosts (1 overlap) → **34/45 admissible-with-penalty; 18 of those are also fully live (resp>=0.4, act<=90d)**.

**Strength vs T_SOLID:** a full grade below. T_SOLID/grade-4 members hold P27-P31 (LTR search ranking, feed ranking with offline-online correlation, semantic search w/ FAISS, 10M-user recsys); the CV borderlines' ceiling is hedged grade-3 recsys. With 156 clean grade>=4 available, **no CV borderline belongs in the top-100**. Place the 34 admissibles in the 100-200 reserve band (MAP insurance if the submission format scores beyond rank 100), ordered by signals, below same-evidence AI-titled grade-3s (title penalty ~ -1 within-band step).

## 2. MAP / tail simulation (ranks 40-100)

Simulator: head fixed at ranks 1-39 (21 live elites + 18 best G4 by C×A); tail policies fill 61 slots; hidden GT simulated as scenario × behavioral-mode, 30 reps of ±1 tier noise (p=0.15); metrics = exact 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP@100 + 0.05·P@10 (gains 2^t−1; relevant = tier>=3; MAP normalized by min(R,100)).

Scenarios: **A180** (R≈180: elites 5, G4 4, 12 strong-G3 3), **B504** (R≈504: + the 336 strong-G3 = P22/P30 holders at tier 3 — exactly round-1's 504 = 168+336, reconciled), **C1068** (all 961 G3 at tier 3). Modes: **avail** (ghosts → tier 2, soft-dead → −1) vs **content** (text-only tiers).

Policies: (a) tail = grade-3 by signals; (b) tail = grade>=4 by content only (6 ghosts land ranks ~40-45); (c) tail = remaining pool by C×A (live G4 outranks ghost-elite; tail = 61 live G4).

| scenario / GT mode | (a) G3-by-signals | (b) G4-content-only | (c) mixed C×A |
|---|---|---|---|
| A180 avail | 0.8382 (−0.1081) | 0.9191 (−0.0272) | **0.9463** |
| A180 content | 0.8255 (−0.1070) | **0.9461** (+0.0136) | 0.9325 |
| B504 avail | 0.8670 (−0.0820) | 0.9223 (−0.0267) | **0.9490** |
| B504 content | 0.8542 (−0.0811) | **0.9490** (+0.0137) | 0.9353 |
| C1068 avail | 0.9248 (−0.0242) | 0.9223 (−0.0267) | **0.9490** |
| C1068 content | 0.9118 (−0.0235) | **0.9490** (+0.0137) | 0.9353 |
| **mean / min / max-regret** | 0.8702 / 0.8255 / 0.1206 | 0.9346 / 0.9191 / 0.0272 | **0.9412 / 0.9325 / 0.0137** |

**Policy (c) is the robust winner** — best mean, best worst-case, half the regret of (b). Policy (a) is dominated everywhere (even in C1068-avail, where grade-3 is relevant, live G4 is still better tail filler). The (b)-vs-(c) gap is entirely the 6 ghosts: (b) wins content cells +0.014, loses avail cells −0.027. Since the envelope evidence (Section 0) says signals ARE tier-conditioned, the avail cells deserve more weight — (c) by a margin.

## 3. Elite-ghost placement (the 6: CAND_0007411/0033861/0041611/0060072/0092278/0094759)

Block-insert into the policy-(c) top-100 at varying ranks; GT ghost-tier ∈ {2 (availability-penalized), 4, 5 (content-dominant)}; averaged over scenarios:

| placement | tier=2 | tier=4 | tier=5 | E[60/30/10] | E[50/30/20] | E[uniform] |
|---|---|---|---|---|---|---|
| ranks 11-16 | 0.9111 | 0.9504 | 0.9569 | 0.9275 | 0.9320 | 0.9396 |
| ranks 31-36 | 0.9255 | 0.9537 | 0.9557 | 0.9370 | 0.9400 | 0.9451 |
| ranks 45-50 | 0.9292 | 0.9536 | 0.9545 | 0.9391 | 0.9416 | 0.9459 |
| ranks 75-80 | 0.9449 | 0.9536 | 0.9417 | 0.9472 | 0.9469 | 0.9467 |
| **ranks 95-100** | 0.9467 | 0.9536 | 0.9417 | 0.9483 | 0.9478 | 0.9473 |
| out of top-100 | 0.9550 | 0.9536 | 0.9417 | 0.9532 | 0.9519 | 0.9500 |

- Asymmetric payoff: placing at 11-16 costs **−0.044** if GT tiers ghosts 2, gains only **+0.015** if tier 5. Ranks 11-30 are ruled out; 30-60 still loses −0.026/−0.020 under tier-2.
- 95-100 vs out-of-100 is a pure bet on P(tier=2): out wins +0.008·P(t2), and they are exactly equal under tier-4/5 (swapped-out live G4s are equally relevant there). Keeping them at 90-100 is cheap insurance against our G4 grades being wrong (if some displaced G4s are actually tier<3, in-list ghosts recover MAP).
- **Recommendation: ranks ~90-100** (or interleave 2 at 85-90, 4 at 93-100). Never above rank 60. This caps worst-case loss at 0.008 while retaining content-dominant upside; consistent with the synthesis hedge "inside 50-100 rather than out", refined to the bottom decile.

## 4. Availability weighting: multiplicative vs additive

Top-100 selected by score over the clean 1117-candidate pool, evaluated over all 6 GT cells:

| scheme | params | mean | min |
|---|---|---|---|
| **multiplicative** S=C·(amin+(1.1−amin)·A01), ghost floor | amin ∈ {0.40, 0.50, 0.60, 0.75} | **0.9404-0.9406** | **0.9326-0.9329** |
| multiplicative, weak | amin = 0.90 | 0.9316 | 0.9128 |
| no availability | — | 0.9326 | 0.9137 |
| additive S=C+w·100·A01 | w ∈ {0.05, 0.10, 0.20} | 0.9315-0.9330 | 0.9102-0.9144 |
| additive | w = 0.30 | 0.9346 | 0.9275 |
| additive | w = 0.50 | 0.9398 | 0.9328 |
| additive + explicit ghost −25 | w ∈ {0.10-0.30} | 0.9401-0.9404 | 0.9324-0.9326 |

- **Multiplicative is flat-optimal across amin 0.40-0.75** — the grade gaps (100/80/60) times any floor ≤0.75 keep live-over-dead ordering intact, so the exact weight doesn't matter. There is a cliff between 0.75 and 0.90 (ghost C·0.9=90 overtakes live G4 ~88 → avail cells crash to 0.91).
- **Additive is fragile**: w ≤ 0.2 leaves ghost elites (C=100, A01≈0.2) above live G4s — it needs w ≥ 0.5 (availability swing = 2.5× a full grade gap) or an explicit ghost-demotion term to match the multiplier.
- **Recommendation: multiplicative C×F×A with availability floor amin ≈ 0.5 (anything in 0.4-0.75 is equivalent) and the hard-ghost floor (resp<0.2 ∧ act>100d ⇒ A=amin).** If the final ranker is additive for other reasons, it must carry a dedicated ghost-demotion term ≥ one full grade gap.

## 5. Within-grade ordering: substructure of the clean 156 (grade>=4)

- Joint GMM on (resp, act): BIC k=2 = 790 vs k=1 = 871 → "bimodal", **but the second mode is exactly the 6 ghosts** (kmeans: cluster of 6 @ resp 0.105 / act 171d vs 150 @ resp 0.685 / act 50d).
- **G4-only is unimodal** (BIC k=1 756.5 < k=2 758.6); a forced 2-split (resp 0.54/act 56 vs 0.815/50) is a continuum cut, and it does **not** align with paragraph identity (P27/P28/P29/P31 crosstab flat).
- The real structure is the **envelope floors** (Section 0): live-elite resp ≥ 0.55, G4 resp ≥ 0.41, G3 resp ≥ 0.25, with a hard gap 0.16→0.41 occupied by nobody in grade>=4.
- **Implication:** there is no hidden tier-4-vs-5 cluster signal inside grade-4 — do not try to promote G4s to tier-5 via signals. Use signals only as (i) the ghost rule, (ii) a smooth within-grade ordering prior. Weak corollary: any grade-4 with resp < 0.55 cannot be inside the tier-5 envelope; the 60 G4s with resp ≥ 0.55 ∧ act ≤ 74 are the only plausible hidden-5 promotions if extra rank-11-30 material is ever needed.

## Policy table (final)

| rank band | fill | ordering |
|---|---|---|
| 1-10 | live elites (21 clean: 13 T_ELITE-live + 8 T_PLAIN) | C×F×A; interleave ELITE/PLAIN per round-2 head analysis |
| 11-39 | remaining live elites + best G4 | C×F×A (multiplier, amin≈0.5) |
| 40-89 | live G4 only (129 clean G4 cover this fully) | C×F×A; within-grade by signals prior |
| 90-100 | last live G4s + **the 6 elite ghosts at ~90-100** | ghosts after live G4, before any grade-3 |
| 100+ reserve (if format scores deeper) | strong-G3 (336 P22/P30 holders) by signals, then other G3, then the **34 admissible CV+P22 borderlines** (title-penalized), then grade-2 | C×A |
| never in top-100 | grade-3-only candidates, CV borderlines, the 14 in-pool honeypots (2 grade-5! 10 grade-4, 2 grade-3 — exclude before assembly) | — |

Worst-case composite of this assembly across all 6 simulated GT cells: 0.948 (avail) / 0.935 (content); no policy variant tested beats it on both mean and min.
