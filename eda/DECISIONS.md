# Design Decisions & Documented Deviations

Decisions where we deliberately diverged from an intermediate recommendation,
with rationale. (Stage-5 defense notes.)

## 1. Elite ghosts excluded from the top-100 (vs "pin at ranks 95-100")

Six candidates carry elite retrieval/ranking career text but dead behavioral
signals (response 7-16%, inactive 114-215 days). The tail simulation's own
winning policy (c) — "live grade-4 over ghost grade-5" — scores best in mean
(0.9412) AND worst-case (0.9325) across all simulated ground truths, and the
marginal value of pinning ghosts at ranks 95-100 was bounded at ±0.008. Our
continuous score (content x availability with a 0.5 ghost floor) places them
just below the rank-100 boundary, which IS policy (c). We kept the principled
continuous rule rather than special-casing six ids: the JD itself says a
perfect-on-paper candidate who ignores recruiters "is, for hiring purposes,
not actually available."

## 2. Formula ordering for ranks 1-15 (vs the hand-pinned ordering)

A deep-read review produced a hand-ordering of the top 15. We encoded its
*principles* (location ladder first, then response rate, recency, notice;
job-hopper demotion; ELITE/PLAIN family hedge) as scoring rules rather than
hardcoding ids. The resulting order agrees with the hand-ordering on the
composition of the head and differs only in within-tier shuffles the review
itself marked contestable (e.g. rank 1: highest-evidence-density Noida
candidate vs highest-response Pune candidate — both grade-5, both in JD
target cities). Hardcoded id lists in scoring code would be indefensible at
code review; rules are auditable and generalize.

## 3. Job-hopper at rank ~23 despite the JD anti-pattern

CAND_0000031 (Hyderabad, 4 strong evidence paragraphs, response 0.91, active
17 days ago) has 3 stints under 20 months. The JD's anti-pattern is
"optimizing for titles by switching companies every 1.5 years" — a
multiplicative 0.88 demotion drops them out of the top-10 but their evidence
plus availability legitimately beats weaker-evidence candidates below. A
recruiter would shortlist them with the tenure question flagged — which is
exactly what our reasoning string does.

## 4. Abroad-without-relocation demoted to effectively out-of-list

The JD hires in India ("outside India: case-by-case, no visa sponsorship").
A London-based candidate with willing_to_relocate=False carried elite text
but cannot realistically be hired; location multiplier for
abroad-no-relocation is 0.42, which places such profiles below the
rank-100 boundary while keeping the rule continuous (a hypothetical
abroad-no-relocate candidate with overwhelming evidence could still
surface — "case-by-case", as the JD says).
