"""Reasoning generation: 1-2 specific, honest sentences per ranked candidate.

Every clause is derived from facts present in the candidate's profile —
their actual career-history paragraphs (via evidence tags supplied in
config, attributed to the employer the work was done at), their signals,
and their logistics. Nothing is invented; concerns are stated when the
underlying gap is real; tone follows rank band. Frame and paraphrase
selection is a deterministic hash of candidate_id so output is reproducible
yet varied.
"""

import hashlib
import math


def _h(cid, salt, n):
    return int(hashlib.md5(f"{cid}:{salt}".encode()).hexdigest(), 16) % n


def _pick(cid, salt, options):
    return options[_h(cid, salt, len(options))]


def _evidence(row, grades, tags):
    """Best-graded paragraph with a tag -> (tag_text, company, is_current)."""
    best = None
    best_grade = -1
    for pp, job in zip(row["para_prefixes"], row["para_jobs"]):
        g = grades.get(pp, 0)
        if pp in tags and g > best_grade:
            best, best_grade = (pp, job), g
    if best is None:
        return None, None, False
    (pp, (company, is_current)) = best
    tag = tags[pp]
    if isinstance(tag, list):
        tag = _pick(row["candidate_id"], "ev", tag)
    return tag, company, is_current


def _evidence_clause(row, grades, tags):
    """Evidence with correct employer attribution."""
    ev, company, is_current = _evidence(row, grades, tags)
    if ev is None:
        return None
    if is_current or company == (row["current_company"] or "").strip():
        return ev
    if company:
        return _pick(row["candidate_id"], "attr", [
            f"at {company}, {ev}",
            f"{ev} (at {company})",
            f"earlier at {company}, {ev}",
        ])
    return f"in an earlier role, {ev}"


def _concerns(row):
    """Noun-phrase concerns, each traceable to a profile fact."""
    out = []
    yoe = row["yoe"]
    if row["location_tier"] == 0:
        out.append(f"based in {row['country']} with no relocation plans (JD hires in India, no visa sponsorship)")
    elif row["location_tier"] == 0.5:
        out.append(f"location ({row['country']}, open to relocation)")
    if row["notice_days"] and row["notice_days"] >= 90:
        out.append(f"a {int(row['notice_days'])}-day notice period")
    dsa = row["days_since_active"]
    if dsa is not None and dsa > 60:
        out.append(f"no platform activity for {int(dsa)} days")
    rr = row["response_rate"] or 0
    if rr < 0.3:
        out.append(f"a low recruiter response rate ({rr:.0%})")
    if not row["open_to_work"] and (dsa is not None and dsa > 60 or rr < 0.5):
        out.append("no open-to-work flag")
    if yoe < 5:
        out.append(f"experience slightly under the 5-9 yr band ({yoe:.1f} yrs)")
    elif yoe > 9:
        out.append(f"experience above the 5-9 yr band ({yoe:.1f} yrs)")
    if row["consulting_only"]:
        out.append("a career entirely at consulting firms, which the JD treats cautiously")
    if row["n_short_stints"] >= 3 and row["avg_tenure_months"] < 20:
        out.append(f"short average tenure (~{row['avg_tenure_months']:.0f} months/role)")
    if row["title_class"] == "CV":
        out.append("a primarily computer-vision background")
    if row["best_grade"] == 3:
        out.append("ML evidence that is adjacent rather than core retrieval/ranking")
    return out


def _strengths(row):
    out = []
    rr = row["response_rate"] or 0
    dsa = row["days_since_active"]
    if rr >= 0.6:
        out.append(f"responds to {rr:.0%} of recruiter outreach")
    if dsa is not None and dsa <= 45:
        wk = max(1, math.ceil(dsa / 7.0))
        out.append(f"active within the last {wk} week{'s' if wk > 1 else ''}")
    if row["notice_days"] is not None and row["notice_days"] <= 30:
        if row["notice_days"] == 0 and row["open_to_work"]:
            out.append("immediately available")
        else:
            out.append(f"{int(row['notice_days'])}-day notice")
    if row["open_to_work"]:
        out.append("open to work")
    if row["location_tier"] == 3:
        out.append(f"based in {row['location'].split(',')[0]}")
    if (row["github_activity"] or -1) >= 50:
        out.append(f"strong public GitHub activity ({row['github_activity']:.0f}/100)")
    if (row["jd_stack_mean"] or 0) >= 60 and row["n_jd_stack_topics"] >= 2:
        out.append(
            f"scored {row['jd_stack_mean']:.0f} average across "
            f"{row['n_jd_stack_topics']} retrieval-stack skill assessments"
        )
    return out


def generate_reasoning(row, rank, grades, tags):
    cid = row["candidate_id"]
    role = f"{row['current_title']} at {row['current_company']}"
    yoe = f"{row['yoe']:.1f} yrs"
    ev = _evidence_clause(row, grades, tags)
    strengths = _strengths(row)
    concerns = _concerns(row)

    s_txt = "; ".join(strengths[:2]) if strengths else ""

    if rank <= 10:
        frames = [
            f"{role} ({yoe}) who {ev} — squarely the JD's production retrieval/ranking profile.",
            f"Exceptional fit: {ev} over a {yoe} career ({row['current_title']}).",
            f"{role}, {yoe}: {ev}.",
        ]
        text = _pick(cid, "f", frames)
        if s_txt:
            text += f" {s_txt[0].upper()}{s_txt[1:]}."
        if concerns:
            text += f" Only caveat: {concerns[0]}."
    elif rank <= 35:
        text = _pick(cid, "f", [
            f"{role} ({yoe}) — {ev}.",
            f"Strong candidate: {ev} ({row['current_title']}, {yoe}).",
        ])
        if s_txt:
            text += f" {s_txt[0].upper()}{s_txt[1:]}."
        if concerns:
            text += f" Watch: {concerns[0]}."
    elif rank <= 70:
        text = f"{role} ({yoe}) — {ev}." if ev else f"{role} ({yoe}) with relevant applied-ML production work."
        joined = "; ".join(concerns[:2])
        if joined:
            text += _pick(cid, "c", [f" Concerns: {joined}.", f" Held back by {joined}."])
        elif s_txt:
            text += f" {s_txt[0].upper()}{s_txt[1:]}."
    else:
        text = _pick(cid, "f", [
            f"{role} ({yoe}) — {ev}." if ev else f"{role} ({yoe}); adjacent ML/data production experience.",
            f"Borderline include: {row['current_title']}, {yoe}; {ev}." if ev else f"Borderline include: {row['current_title']}, {yoe}.",
        ])
        joined = "; ".join(concerns[:2])
        if joined:
            text += f" Ranked low given {joined}."
        elif s_txt:
            text += f" Still, {s_txt}."
    return " ".join(text.split())
