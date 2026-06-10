import json, re, sys
from datetime import date

JSONL = r"C:\Users\yosoy\OneDrive\Desktop\Github\Hackathons\hack2skill\The Data & AI Challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
PARAS = json.load(open(r"C:\Users\yosoy\OneDrive\Desktop\Github\Hackathons\redrob-ranker\eda\round2\paragraphs_44.json", encoding="utf-8"))
GRADES = {p["pid"]: g["grade"] for p, g in zip(PARAS, json.load(open(r"C:\Users\yosoy\OneDrive\Desktop\Github\Hackathons\redrob-ranker\eda\round2\paragraph_grades_consensus.json", encoding="utf-8")))}
PREF = {p["text"][:60]: p["pid"] for p in PARAS}

ids = set(json.load(open(r"C:\Users\yosoy\OneDrive\Desktop\Github\Hackathons\redrob-ranker\eda\round2\_top_union_ids.json")))
batch = sys.argv[1] if len(sys.argv) > 1 else "all"

REF = date(2026, 6, 10)

def months_between(s, e):
    try:
        sy, sm = int(s[:4]), int(s[5:7])
    except Exception:
        return None
    if e in (None, "", "present", "Present"):
        ey, em = REF.year, REF.month
    else:
        try:
            ey, em = int(e[:4]), int(e[5:7])
        except Exception:
            return None
    return (ey - sy) * 12 + (em - sm)

def pid_of(desc):
    return PREF.get((desc or "")[:60], "??")

recs = {}
with open(JSONL, encoding="utf-8") as f:
    for line in f:
        m = re.search(r'"candidate_id":\s*"(CAND_\d+)"', line[:200])
        if m and m.group(1) in ids:
            recs[m.group(1)] = json.loads(line)
            if len(recs) == len(ids):
                break

order = sorted(recs)
if batch == "A":
    order = order[:15]
elif batch == "B":
    order = order[15:]

YOE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", re.I)

for cid in order:
    r = recs[cid]
    p = r.get("profile") or {}
    print("=" * 100)
    print(f"### {cid} | {p.get('anonymized_name')} | {p.get('headline')}")
    print(f"loc: {p.get('location')}, {p.get('country')} | yoe_field: {p.get('years_of_experience')} | {p.get('current_title')} @ {p.get('current_company')} ({p.get('current_company_size')}, {p.get('current_industry')})")
    print(f"SUMMARY: {p.get('summary')}")
    jobs = r.get("career_history") or []
    spans = []
    hpB = []
    for j in jobs:
        s, e = j.get("start_date"), j.get("end_date")
        dm = j.get("duration_months")
        mb = months_between(s, e)
        if mb is not None:
            spans.append((s, e, mb))
        gap = None if (mb is None or dm is None) else abs(dm - mb)
        if gap is not None and gap > 3:
            hpB.append((j.get("title"), dm, mb, gap))
        pid = pid_of(j.get("description"))
        g = GRADES.get(pid, "?")
        print(f"  JOB: {j.get('title')} @ {j.get('company')} ({j.get('company_size','?')}, {j.get('industry','?')}) {s} -> {e} dur={dm}mo calc={mb}mo | para={pid}(g{g})")
        print(f"       desc: {(j.get('description') or '')[:170]}")
    edus = r.get("education") or []
    for ed in edus:
        print(f"  EDU: {ed.get('degree')} {ed.get('field_of_study')} @ {ed.get('institution')} (tier {ed.get('tier')}) {ed.get('start_year')}-{ed.get('end_year')}")
    skills = r.get("skills") or []
    sk = "; ".join(f"{s.get('name')}[{s.get('proficiency')},{s.get('duration_months')}mo]" for s in skills)
    print(f"  SKILLS({len(skills)}): {sk}")
    certs = r.get("certifications") or []
    print(f"  CERTS: {[(c.get('name'), c.get('year')) for c in certs]}")

    # honeypot markers
    A = sum(1 for s in skills if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0)
    starts = [s for s, e, mb in spans]
    ends = [(e if e not in (None, "", "present", "Present") else "2026-06") for s, e, mb in spans]
    span_m = None
    if starts:
        span_m = months_between(min(starts), max(ends))
    yoe = p.get("years_of_experience") or 0
    C = (span_m is not None) and (yoe * 12 > span_m + 24)
    m = YOE_RE.search(p.get("summary") or "")
    sum_yoe = float(m.group(1)) if m else None
    D = (sum_yoe is not None) and abs(sum_yoe - yoe) > 1.5
    E = [c.get("name") for c in certs if (c.get("year") or 0) > 2026]
    print(f"  HONEYPOT: A_expert0mo={A} | B_durMismatch={hpB} | C span={span_m}mo yoe*12={yoe*12:.0f} -> {C} | D sum_yoe={sum_yoe} vs {yoe} -> {D} | E futureCerts={E}")
    n_jobs = len(jobs)
    short = sum(1 for s, e, mb in spans if mb is not None and mb < 18 and e not in (None, "", "present", "Present"))
    print(f"  CAREER: n_jobs={n_jobs} short_stints(<18mo,ended)={short} companies={[j.get('company') for j in jobs]}")
