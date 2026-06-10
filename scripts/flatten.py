#!/usr/bin/env python3
"""Flatten candidates.jsonl into a single parquet table for fast analysis.

One row per candidate. Nested structures are aggregated into scalar features;
long-text fields are preserved so downstream steps can embed or inspect them.
"""

import argparse
import json
from datetime import date

import pandas as pd

REFERENCE_DATE = date(2026, 6, 10)  # analysis date; only used for derived deltas


def parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def months_between(d1, d2):
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def flatten(c):
    p = c.get("profile", {})
    jobs = c.get("career_history", []) or []
    edu = c.get("education", []) or []
    skills = c.get("skills", []) or []
    certs = c.get("certifications", []) or []
    langs = c.get("languages", []) or []
    sig = c.get("redrob_signals", {}) or {}

    row = {
        "candidate_id": c.get("candidate_id"),
        # profile
        "name": p.get("anonymized_name"),
        "headline": p.get("headline"),
        "summary": p.get("summary"),
        "location": p.get("location"),
        "country": p.get("country"),
        "yoe": p.get("years_of_experience"),
        "current_title": p.get("current_title"),
        "current_company": p.get("current_company"),
        "current_company_size": p.get("current_company_size"),
        "current_industry": p.get("current_industry"),
    }

    # career history aggregates
    durations = [j.get("duration_months") or 0 for j in jobs]
    starts = [parse_date(j.get("start_date")) for j in jobs]
    ends = [parse_date(j.get("end_date")) for j in jobs]
    spans = []  # (start, end) with None end -> reference date
    date_duration_mismatch = 0
    for j, s, e in zip(jobs, starts, ends):
        eff_end = e or REFERENCE_DATE
        if s:
            span_months = months_between(s, eff_end)
            spans.append((s, eff_end))
            claimed = j.get("duration_months")
            if claimed is not None and abs(span_months - claimed) > 3:
                date_duration_mismatch += 1

    # overlap months between consecutive jobs (sorted by start)
    overlap_months = 0
    sorted_spans = sorted([sp for sp in spans if sp[0]], key=lambda x: x[0])
    for a, b in zip(sorted_spans, sorted_spans[1:]):
        if b[0] < a[1]:
            overlap_months += months_between(b[0], min(a[1], b[1]))

    career_start = min((s for s in starts if s), default=None)
    career_span_months = months_between(career_start, REFERENCE_DATE) if career_start else None

    row.update({
        "n_jobs": len(jobs),
        "total_job_months": sum(durations),
        "career_span_months": career_span_months,
        "career_start": str(career_start) if career_start else None,
        "n_current_jobs": sum(1 for j in jobs if j.get("is_current")),
        "avg_tenure_months": (sum(durations) / len(jobs)) if jobs else None,
        "n_short_stints": sum(1 for d in durations if d < 18),
        "overlap_months": overlap_months,
        "date_duration_mismatch": date_duration_mismatch,
        "job_titles": " | ".join((j.get("title") or "") for j in jobs),
        "job_companies": " | ".join((j.get("company") or "") for j in jobs),
        "job_industries": " | ".join((j.get("industry") or "") for j in jobs),
        "job_descriptions": "\n".join((j.get("description") or "") for j in jobs),
    })

    # education
    tiers = [e.get("tier") for e in edu if e.get("tier")]
    row.update({
        "n_edu": len(edu),
        "best_edu_tier": min(tiers, default=None),  # tier_1 < tier_2 ... lexicographic works
        "edu_institutions": " | ".join((e.get("institution") or "") for e in edu),
        "edu_degrees": " | ".join((e.get("degree") or "") for e in edu),
        "edu_fields": " | ".join((e.get("field_of_study") or "") for e in edu),
    })

    # skills
    prof_rank = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    row.update({
        "n_skills": len(skills),
        "skill_names": " | ".join((s.get("name") or "") for s in skills),
        "n_expert": sum(1 for s in skills if s.get("proficiency") == "expert"),
        "n_advanced": sum(1 for s in skills if s.get("proficiency") == "advanced"),
        "n_expert_zero_months": sum(
            1 for s in skills
            if prof_rank.get(s.get("proficiency"), 0) >= 4 and (s.get("duration_months") or 0) == 0
        ),
        "total_skill_endorsements": sum(s.get("endorsements") or 0 for s in skills),
        "skills_json": json.dumps(skills),
        "n_certs": len(certs),
        "cert_names": " | ".join((cc.get("name") or "") for cc in certs),
        "n_langs": len(langs),
    })

    # redrob signals — flatten all
    sal = sig.get("expected_salary_range_inr_lpa") or {}
    assess = sig.get("skill_assessment_scores") or {}
    row.update({
        "sig_profile_completeness": sig.get("profile_completeness_score"),
        "sig_signup_date": sig.get("signup_date"),
        "sig_last_active_date": sig.get("last_active_date"),
        "sig_open_to_work": sig.get("open_to_work_flag"),
        "sig_profile_views_30d": sig.get("profile_views_received_30d"),
        "sig_applications_30d": sig.get("applications_submitted_30d"),
        "sig_response_rate": sig.get("recruiter_response_rate"),
        "sig_avg_response_hours": sig.get("avg_response_time_hours"),
        "sig_n_assessments": len(assess),
        "sig_assessment_mean": (sum(assess.values()) / len(assess)) if assess else None,
        "sig_assessments_json": json.dumps(assess),
        "sig_connection_count": sig.get("connection_count"),
        "sig_endorsements_received": sig.get("endorsements_received"),
        "sig_notice_period_days": sig.get("notice_period_days"),
        "sig_salary_min": sal.get("min"),
        "sig_salary_max": sal.get("max"),
        "sig_work_mode": sig.get("preferred_work_mode"),
        "sig_willing_to_relocate": sig.get("willing_to_relocate"),
        "sig_github_activity": sig.get("github_activity_score"),
        "sig_search_appearance_30d": sig.get("search_appearance_30d"),
        "sig_saved_by_recruiters_30d": sig.get("saved_by_recruiters_30d"),
        "sig_interview_completion": sig.get("interview_completion_rate"),
        "sig_offer_acceptance": sig.get("offer_acceptance_rate"),
        "sig_verified_email": sig.get("verified_email"),
        "sig_verified_phone": sig.get("verified_phone"),
        "sig_linkedin_connected": sig.get("linkedin_connected"),
    })

    # derived recency
    last_active = parse_date(sig.get("last_active_date"))
    signup = parse_date(sig.get("signup_date"))
    row["days_since_active"] = (REFERENCE_DATE - last_active).days if last_active else None
    row["signup_after_active"] = bool(signup and last_active and signup > last_active)

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.inp, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                rows.append(flatten(json.loads(line)))
            if (i + 1) % 20000 == 0:
                print(f"  {i + 1} processed")

    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols -> {args.out}")


if __name__ == "__main__":
    main()
