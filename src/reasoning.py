"""Reasoning generation: 1-2 specific, honest sentences per ranked candidate.

Every clause is derived from facts present in the candidate's profile —
their actual career-history paragraphs (via evidence tags for the graded
paragraph fingerprints, supplied in config), their signals, and their
logistics. Nothing is invented; concerns are stated when the underlying gap
is real; tone follows rank band. Frame selection is a deterministic hash of
candidate_id so output is reproducible yet varied.
"""

import hashlib


def _h(cid, salt, n):
    return int(hashlib.md5(f"{cid}:{salt}".encode()).hexdigest(), 16) % n


def _pick(cid, salt, options):
    return options[_h(cid, salt, len(options))]


def _evidence(row):
    best, best_grade = None, -1
    grades = row["_grades_by_prefix"]
    tags = row["_evidence_tags"]
    for pp in row["para_prefixes"]:
        g = grades.get(pp, 0)
        if pp in tags and g > best_grade:
            best, best_grade = pp, g
    return tags.get(best)


def _concerns(row):
    out = []
    if row["notice_days"] and row["notice_days"] >= 90:
        out.append(f"{int(row['notice_days'])}-day notice period")
    dsa = row["days_since_active"]
    if dsa is not None and dsa > 90:
        out.append(f"last active ~{int(round(dsa / 30.0))} months ago")
    rr = row["response_rate"] or 0
    if rr < 0.3:
        out.append(f"low recruiter response rate ({rr:.0%})")
    if row["location_tier"] == 0.5:
        out.append(f"based in {row['country']} (open to relocation)")
    if row["agreed_yoe"] < 5:
        out.append(f"slightly below the 5-9 yr band at {row['agreed_yoe']:.1f} yrs")
    elif row["agreed_yoe"] > 9:
        out.append(f"above the 5-9 yr band at {row['agreed_yoe']:.1f} yrs")
    if row["consulting_only"]:
        out.append("career entirely at consulting firms, which the JD treats cautiously")
    if row["title_class"] == "CV":
        out.append("primarily a computer-vision background")
    if row["best_grade"] == 3:
        out.append("ML evidence is adjacent rather than core retrieval/ranking")
    return out


def _strengths(row):
    out = []
    rr = row["response_rate"] or 0
    dsa = row["days_since_active"]
    if rr >= 0.6:
        out.append(f"responds to {rr:.0%} of recruiter outreach")
    if dsa is not None and dsa <= 45:
        wk = max(1, int(round(dsa / 7.0)))
        out.append(f"active within the last {wk} week{'s' if wk > 1 else ''}")
    if row["notice_days"] is not None and row["notice_days"] <= 30:
        out.append(f"{int(row['notice_days'])}-day notice" if row["notice_days"] else "immediately available")
    if row["open_to_work"]:
        out.append("open to work")
    if row["location_tier"] == 3:
        out.append(f"based in {row['location'].split(',')[0]}")
    if (row["github_activity"] or -1) >= 50:
        out.append(f"strong public GitHub activity ({row['github_activity']:.0f}/100)")
    if (row["jd_stack_mean"] or 0) >= 60 and row["n_jd_stack_topics"] >= 2:
        out.append(f"scored {row['jd_stack_mean']:.0f} avg on retrieval-stack skill assessments")
    return out


def generate_reasoning(row, rank):
    cid = row["candidate_id"]
    role = f"{row['current_title']} at {row['current_company']}"
    yoe = f"{row['yoe']:.1f} yrs"
    ev = _evidence(row)
    strengths = _strengths(row)
    concerns = _concerns(row)

    s_txt = _pick(cid, "s", [
        "; ".join(strengths[:2]),
        " and ".join(strengths[:2]),
    ]) if strengths else ""

    if rank <= 10:
        frames = [
            f"{role} ({yoe}) who {ev} — squarely the JD's production retrieval/ranking profile. {s_txt.capitalize()}." if s_txt else f"{role} ({yoe}) who {ev} — squarely the JD's production retrieval/ranking profile.",
            f"Exceptional fit: {ev} over a {yoe} career ({row['current_title']}). {s_txt.capitalize()}." if s_txt else f"Exceptional fit: {ev} over a {yoe} career ({row['current_title']}).",
            f"{role}, {yoe}: {ev}. {s_txt.capitalize()}." if s_txt else f"{role}, {yoe}: {ev}.",
        ]
        text = _pick(cid, "f", frames)
        if concerns:
            text += f" Only caveat: {concerns[0]}."
    elif rank <= 35:
        text = _pick(cid, "f", [
            f"{role} ({yoe}) — {ev}.",
            f"Strong candidate: {ev} ({row['current_title']}, {yoe}).",
        ])
        if s_txt:
            text += f" {s_txt.capitalize()}."
        if concerns:
            text += f" Watch: {concerns[0]}."
    elif rank <= 70:
        text = f"{role} ({yoe}) — {ev}." if ev else f"{role} ({yoe}) with relevant applied-ML production work."
        joined = "; ".join(concerns[:2])
        if joined:
            text += _pick(cid, "c", [f" Concerns: {joined}.", f" Held back by {joined}."])
        elif s_txt:
            text += f" {s_txt.capitalize()}."
    else:
        text = _pick(cid, "f", [
            f"{role} ({yoe}) — {ev}." if ev else f"{role} ({yoe}); adjacent ML/data production experience.",
            f"Borderline include: {row['current_title']}, {yoe}; {ev}." if ev else f"Borderline include: {row['current_title']}, {yoe}.",
        ])
        joined = "; ".join(concerns[:2])
        if joined:
            text += f" Ranked low on {joined}."
    return " ".join(text.split())
