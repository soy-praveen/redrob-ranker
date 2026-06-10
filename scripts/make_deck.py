#!/usr/bin/env python3
"""Render the submission deck (PDF) following the organizer template structure."""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INK = "#1a1a2e"
ACCENT = "#c0392b"
MUTED = "#5a6472"
BG = "#ffffff"

W, H = 13.333, 7.5  # 16:9


def slide(pdf, title, blocks, footer=None, title_size=30):
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.045, 0.895), 0.012, 0.062, color=ACCENT))
    ax.text(0.075, 0.925, title, fontsize=title_size, fontweight="bold",
            color=INK, va="center", family="DejaVu Sans")
    y = 0.83
    for kind, text in blocks:
        if kind == "h":
            y -= 0.015
            ax.text(0.075, y, text, fontsize=16, fontweight="bold", color=ACCENT,
                    va="top", family="DejaVu Sans")
            y -= 0.055
        elif kind == "b":
            ax.text(0.085, y, "•", fontsize=13, color=ACCENT, va="top")
            ax.text(0.105, y, text, fontsize=13, color=INK, va="top",
                    family="DejaVu Sans", wrap=True)
            y -= 0.052 * (1 + text.count("\n"))
        elif kind == "t":
            ax.text(0.075, y, text, fontsize=13.5, color=INK, va="top",
                    family="DejaVu Sans")
            y -= 0.052 * (1 + text.count("\n"))
        elif kind == "m":  # monospace block
            ax.text(0.085, y, text, fontsize=11.5, color=INK, va="top",
                    family="DejaVu Sans Mono",
                    bbox=dict(boxstyle="round,pad=0.6", fc="#f4f4f8", ec="#d8d8e0"))
            y -= 0.052 * (1 + text.count("\n")) + 0.04
        elif kind == "gap":
            y -= float(text)
    if footer:
        ax.text(0.075, 0.045, footer, fontsize=10, color=MUTED, va="center")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def main(out="docs/deck.pdf"):
    pdf = PdfPages(out)

    # 1 — Title
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0.045, 0.62), 0.012, 0.13, color=ACCENT))
    ax.text(0.075, 0.72, "Ranking Candidates the Way", fontsize=34, fontweight="bold", color=INK)
    ax.text(0.075, 0.645, "a Great Recruiter Would", fontsize=34, fontweight="bold", color=INK)
    ax.text(0.075, 0.55, "The Data & AI Challenge — Intelligent Candidate Discovery & Ranking",
            fontsize=16, color=MUTED)
    ax.text(0.075, 0.40, "Team Name :  FILL_ME", fontsize=15, color=INK)
    ax.text(0.075, 0.35, "Team Leader :  FILL_ME", fontsize=15, color=INK)
    ax.text(0.075, 0.30, "Problem Statement :  Rank 100,000 candidate profiles against the "
            "Senior AI Engineer JD;\ndeliver a trusted, explainable top-100 shortlist.",
            fontsize=15, color=INK, va="top")
    pdf.savefig(fig, facecolor=BG); plt.close(fig)

    # 2 — Solution Overview
    slide(pdf, "Solution Overview", [
        ("t", "Evidence-graded recruiter model:  final score  =  Content × Fit × Availability + Prior"),
        ("gap", "0.02"),
        ("b", "Content — what the candidate has actually built, read from career-history text against a\ngraded evidence rubric (production retrieval / ranking / recsys highest)."),
        ("b", "Fit — the JD's explicit rules: 5–9 yr band, Pune/Noida-first location ladder,\nconsulting-only and job-hopper demotions."),
        ("b", "Availability — response rate, activity recency, notice period: a perfect profile that\nignores recruiters is not a real candidate (the JD says exactly this)."),
        ("b", "Prior — recruiter-demand signals: search appearances, recruiter saves, verified\nretrieval-stack assessments, public GitHub activity."),
        ("h", "What differentiates it"),
        ("b", "Skills lists and self-summaries earn ZERO positive score — they are where fake profiles\nconcentrate keywords. Career history is the only field that can't be cheaply faked."),
        ("b", "Five profile-consistency checks exclude impossible profiles (the dataset's honeypots)\nbefore scoring; every ranking decision is rule-traceable and auditable."),
    ], footer="Redrob Ranker — Solution Overview")

    # 3 — JD Understanding
    slide(pdf, "JD Understanding & Candidate Evaluation", [
        ("h", "What the JD actually asks for (beyond the keyword list)"),
        ("b", "Must-haves: production embeddings/retrieval experience, vector-search operations,\nranking evaluation (NDCG/MRR, offline-to-online), strong Python — shipped to real users."),
        ("b", "Explicit anti-patterns: research-only careers, <12-month LangChain-wrapper experience,\nconsulting-firm-only careers, CV-only without NLP/IR, title-chasing job-hoppers."),
        ("b", "Logistics matter: Pune/Noida preferred (Hyd/Mum/NCR welcome), sub-30-day notice loved,\nbehavioral availability explicitly called out."),
        ("h", "Signals ranked by importance in our evaluation"),
        ("b", "1. Career-history evidence of retrieval/ranking systems in production  (decisive)"),
        ("b", "2. Behavioral availability: recruiter response rate, last-active recency, notice period"),
        ("b", "3. JD fit: experience band, location/relocation, employer-type history"),
        ("b", "4. Corroboration: skill durations & endorsements, assessment scores, GitHub activity"),
        ("b", "Keyword presence in skills/headline:  weight ZERO — by design."),
    ], footer="Redrob Ranker — JD Understanding")

    # 4 — Ranking Methodology
    slide(pdf, "Ranking Methodology", [
        ("m", "candidates.jsonl ─> features (55) ─> hard exclusions ─> C × F × A + P ─> top-100 + reasoning"),
        ("b", "Retrieve: single streaming pass extracts 55 features per candidate — career evidence,\nconsistency markers, skill-trust metrics, 23 behavioral signals, logistics."),
        ("b", "Score: graded evidence rubric over career-history text (graded against the JD, then\nblind re-graded by a 3-judge panel; disagreements reconciled by median)."),
        ("b", "Skill claims count only when corroborated: long durations, real endorsements, matching\nassessment scores. Uncorroborated keyword lists contribute nothing."),
        ("b", "Combine: content score scaled by two multipliers (fit, availability) plus a small additive\nengagement prior. Multiplicative availability chosen via simulation — additive weighting\nlets behaviorally-dead profiles crowd out hireable ones."),
        ("b", "Heuristics over models at rank time: deterministic rules + pandas/regex. No model\ninference, no embeddings, no network. 100k candidates in ~15 seconds on CPU."),
    ], footer="Redrob Ranker — Methodology")

    # 5 — Explainability & Data Validation
    slide(pdf, "Explainability & Data Validation", [
        ("h", "Explainability"),
        ("b", "Every candidate's reasoning is generated from their own profile facts: strongest career\nevidence, concrete signal values (response %, notice days), and honest concerns."),
        ("b", "No claim can appear that is not in the profile — the generator only composes from\nextracted fields, so unsupported justifications are structurally impossible."),
        ("b", "Tone follows rank: top picks read strong; rank-90s say honestly why they are borderline."),
        ("h", "Suspicious-profile handling (the planted traps)"),
        ("b", "Honeypots: 5 independent impossibility checks (expert skills with 0 months; durations\ncontradicting dates; experience > career span; summary vs field mismatch; future certs)."),
        ("b", "Keyword stuffers: non-technical titles carrying 7–13 AI skills with zero ML evidence\nin career history — excluded; cloned skill-lists deduplicated to the coherent member."),
        ("b", "Ghosts: elite text + dead signals (response <20%, inactive >100d) — hard-demoted."),
        ("b", "Final assembly re-asserts: zero honeypot markers in the top-100, zero stuffers,\nmonotone scores, validator-compliant tie-breaks."),
    ], footer="Redrob Ranker — Explainability & Validation")

    # 6 — End-to-End Workflow
    slide(pdf, "End-to-End Workflow", [
        ("m", "JD analysis (one-time, human + AI-assisted)\n"
              "   └─ graded evidence rubric + JD fit rules ──> config/params.json\n\n"
              "python rank.py --candidates candidates.jsonl --out submission.csv\n"
              "   1. stream-parse 100k profiles            (~10 s)\n"
              "   2. derive 55 features per candidate       (memoized text analysis)\n"
              "   3. hard exclusions                        (~31k profiles removed)\n"
              "   4. score C x F x A + P, sort, tie-break\n"
              "   5. generate per-candidate reasoning\n"
              "   6. assembly assertions + write CSV        (~15 s total)"),
        ("b", "One command, deterministic output, no pre-computation step required."),
        ("b", "Sandbox (Streamlit) runs the identical pipeline on any uploaded sample."),
    ], footer="Redrob Ranker — Workflow")

    # 7 — System Architecture
    slide(pdf, "System Architecture", [
        ("m", "rank.py (CLI)\n"
              "  ├── src/features.py    streaming JSONL -> 55-feature table\n"
              "  │     • title taxonomy, evidence lexicons, consistency markers\n"
              "  │     • memoized text scans (templated corpus -> ~1000x fewer regex calls)\n"
              "  ├── src/scoring.py     exclusions -> C x F x A + P -> ranked 100\n"
              "  │     • all weights/grades externalized to config/params.json\n"
              "  ├── src/reasoning.py   profile facts -> recruiter-style explanation\n"
              "  └── config/params.json the full rubric: 44 graded evidence patterns,\n"
              "                         fit ladders, availability steps, prior weights"),
        ("b", "Pure pandas + numpy + regex. No service dependencies, no GPU, no network, < 4 GB RAM."),
        ("b", "Same code path powers the CLI and the hosted sandbox."),
    ], footer="Redrob Ranker — Architecture")

    # 8 — Results & Performance
    slide(pdf, "Results & Performance", [
        ("h", "Ranking quality"),
        ("b", "Top-100 is 100% candidates with production-ML evidence in career history (99% India,\nall within the JD's experience band after consistency-checked years-of-experience)."),
        ("b", "Catches all 8 'plain-language' senior ranking engineers (zero buzzwords in their\nprofiles) in the head of the list — the trap naive keyword/embedding rankers miss."),
        ("b", "Zero honeypots in the top-100 (verified by independent re-derivation from raw data\nin an adversarial audit); keyword stuffers score zero by construction."),
        ("b", "Reasoning passes all six Stage-4 checks by construction: specific, JD-connected,\nhonest about gaps, hallucination-free, varied, rank-consistent."),
        ("h", "Compute (constraint: 5 min / 16 GB / CPU-only / no network)"),
        ("b", "~15 seconds end-to-end for 100,000 candidates on a laptop CPU — 20x headroom."),
        ("b", "< 4 GB peak memory, zero network calls, deterministic across runs."),
    ], footer="Redrob Ranker — Results")

    # 9 — Technologies Used
    slide(pdf, "Technologies Used", [
        ("b", "Python 3.13, pandas, numpy — the entire ranking pipeline. Chosen for determinism,\nauditability, and CPU speed: the task is evidence reading, not similarity search."),
        ("b", "Regex evidence lexicons + graded rubric (config/params.json) — transparent,\nreviewable scoring; every rank is explainable line-by-line."),
        ("b", "Streamlit — hosted sandbox running the identical pipeline on uploaded samples."),
        ("b", "Claude (AI-assisted engineering, declared) — EDA at scale, blind multi-judge rubric\ngrading, adversarial QA. All outputs verified by re-derivation from raw data."),
        ("h", "Why NOT embeddings / LLM ranking at inference"),
        ("b", "Recruiting corpora are adversarial: keyword-stuffed profiles and near-duplicate texts\nreward surface similarity. Embedding rankers score the traps highly, miss plain-language\nevidence, and cost ~1000x more compute. Reading structured evidence wins on quality,\nspeed, explainability — and scales to 200k+ candidates in production."),
    ], footer="Redrob Ranker — Technologies")

    # 10 — Submission Assets
    slide(pdf, "Submission Assets", [
        ("b", "GitHub repository:  https://github.com/FILL_ME/redrob-ranker"),
        ("b", "Hosted sandbox (Streamlit):  https://FILL_ME.streamlit.app"),
        ("b", "Ranked output:  submission.csv  (top-100, validator-clean)"),
        ("b", "Reproduce:  python rank.py --candidates ./candidates.jsonl --out ./submission.csv"),
        ("b", "Methodology & analysis:  README.md + eda/ (two analysis rounds, 3-judge rubric\ncalibration, adversarial audit of the final list — full provenance for every parameter)"),
        ("gap", "0.05"),
        ("t", "Built to be defended: every scoring rule traces to a sentence in the JD or a\nmeasured property of the data."),
    ], footer="Redrob Ranker — Assets")

    pdf.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
