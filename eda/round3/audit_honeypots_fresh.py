# Fresh, independent honeypot audit of the 100 submitted candidates.
# Parses RAW JSONL only. Does NOT import anything from src/.
import json, re, csv, sys
from datetime import date

RAW = r"C:\Users\yosoy\OneDrive\Desktop\Github\Hackathons\hack2skill\The Data & AI Challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
SUB = r"C:\Users\yosoy\OneDrive\Desktop\Github\Hackathons\redrob-ranker\submission_draft.csv"
HP  = r"C:\Users\yosoy\OneDrive\Desktop\Github\Hackathons\redrob-ranker\eda\round2\honeypot_final.csv"
TODAY = date(2026, 6, 10)

# ---------- load submission ids ----------
sub_ids = []
with open(SUB, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        sub_ids.append(row["candidate_id"])
sub_set = set(sub_ids)
print(f"submission ids: {len(sub_ids)} (unique {len(sub_set)})")

# ---------- load known honeypot ids ----------
hp_ids = set()
with open(HP, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        hp_ids.add(row["candidate_id"])
print(f"known honeypot ids: {len(hp_ids)}")
overlap = sub_set & hp_ids
print(f"OVERLAP with 93-id honeypot list: {sorted(overlap) if overlap else 'NONE'}")

# ---------- helpers ----------
def parse_date(s):
    if not s:
        return None
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None

def months_between(d1, d2):
    """calendar months with day fraction"""
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) + (d2.day - d1.day) / 30.44

# summary yoe regexes (try several variants, take first hit)
YOE_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+of\s+(?:\w+[- ]){0,3}?experience", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:hands[- ]on\s+)?experience", re.I),
    re.compile(r"experience\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", re.I),  # last resort: any "X yrs"
]

def summary_yoe(text):
    if not text:
        return None, None
    for i, pat in enumerate(YOE_PATTERNS):
        m = pat.search(text)
        if m:
            return float(m.group(1)), i
    return None, None

AI_TITLE_RE = re.compile(
    r"machine learning|ml engineer|\bai\b|artificial intelligence|data scien|nlp|"
    r"applied scien|deep learning|computer vision|recommendation|search engineer|"
    r"applied ml|mlops|research (scientist|engineer)|llm", re.I)

# ---------- single pass over raw jsonl ----------
records = {}                 # our 100 full records
skillset_map = {}            # frozenset(skills) -> list[(cid, title)]
n_lines = 0
with open(RAW, encoding="utf-8") as f:
    for line in f:
        n_lines += 1
        rec = json.loads(line)
        cid = rec["candidate_id"]
        skills = rec.get("skills") or []
        names = frozenset(s.get("name", "") for s in skills)
        title = (rec.get("profile") or {}).get("current_title") or ""
        if names:
            skillset_map.setdefault(names, []).append((cid, title))
        if cid in sub_set:
            records[cid] = rec
print(f"raw lines: {n_lines}; matched submission records: {len(records)}")
missing = sub_set - set(records)
if missing:
    print(f"MISSING FROM RAW: {sorted(missing)}")

# ---------- marker evaluation ----------
results = {}
for cid in sub_ids:
    rec = records[cid]
    prof = rec.get("profile") or {}
    yoe = prof.get("years_of_experience")
    career = rec.get("career_history") or []
    skills = rec.get("skills") or []
    certs = rec.get("certifications") or []
    flags = []
    detail = []

    # A: expert skill with 0 months
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months") == 0:
            flags.append("A")
            detail.append(f"A: skill '{s.get('name')}' expert/0mo")
            break

    # B: job duration mismatch > 3 months
    for j in career:
        sd = parse_date(j.get("start_date"))
        ed = parse_date(j.get("end_date")) or TODAY
        dm = j.get("duration_months")
        if sd is None or dm is None:
            continue
        diff = abs(dm - months_between(sd, ed))
        if diff > 3:
            flags.append("B")
            detail.append(f"B: {j.get('company')} dur={dm}mo vs dates {j.get('start_date')}..{j.get('end_date')} -> diff {diff:.1f}mo")
    # C: yoe*12 > span+24
    starts = [parse_date(j.get("start_date")) for j in career]
    starts = [s for s in starts if s]
    if starts and yoe is not None:
        span = months_between(min(starts), TODAY)
        if yoe * 12 > span + 24:
            flags.append("C")
            detail.append(f"C: yoe={yoe} ({yoe*12:.0f}mo) vs span {span:.1f}mo (+24 allowance)")
    # D: summary yoe gap > 1.5
    s_yoe, pat_idx = summary_yoe(prof.get("summary"))
    if s_yoe is not None and yoe is not None:
        gap = abs(s_yoe - yoe)
        if gap > 1.5:
            flags.append("D")
            detail.append(f"D: summary says {s_yoe} yrs (pattern {pat_idx}) vs field {yoe} -> gap {gap:.1f}")
    elif s_yoe is None:
        detail.append("D: no yoe parsed from summary (no flag)")
    # E: cert year > 2026
    for c in certs:
        yr = c.get("year")
        if yr is not None and yr > 2026:
            flags.append("E")
            detail.append(f"E: cert '{c.get('name')}' year {yr}")

    results[cid] = (sorted(set(flags)), detail, s_yoe, yoe)

flagged = {cid: v for cid, v in results.items() if v[0]}
print(f"\n=== HARD MARKER HITS among our 100: {len(flagged)} ===")
for cid, (fl, det, s_yoe, yoe) in flagged.items():
    print(f"{cid}: {fl}")
    for d in det:
        print(f"   {d}")

# ---------- summary-yoe parse coverage + near misses ----------
print("\n=== D near-miss table (gap > 0.5) and parse failures ===")
for cid in sub_ids:
    fl, det, s_yoe, yoe = results[cid]
    if s_yoe is None:
        print(f"{cid}: summary yoe UNPARSED (field yoe={yoe})")
    elif yoe is not None and abs(s_yoe - yoe) > 0.5:
        print(f"{cid}: summary {s_yoe} vs field {yoe} gap {abs(s_yoe-yoe):.2f}")

# ---------- clone / duplicate skill-set check ----------
print("\n=== duplicate skill-set involvement ===")
dup_count = 0
for cid in sub_ids:
    rec = records[cid]
    names = frozenset(s.get("name", "") for s in (rec.get("skills") or []))
    members = skillset_map.get(names, [])
    if len(members) > 1:
        dup_count += 1
        our_title = (rec.get("profile") or {}).get("current_title") or ""
        our_ai = bool(AI_TITLE_RE.search(our_title))
        others = [(c, t, bool(AI_TITLE_RE.search(t))) for c, t in members if c != cid]
        print(f"{cid} (title='{our_title}', ai={our_ai}) shares exact skill-set with: {others}")
if dup_count == 0:
    print("none of our 100 share an exact unordered skill-set with any other candidate")

# ---------- signup > last_active + other anomalies ----------
print("\n=== signup_date > last_active_date among our 100 ===")
sig_anom = []
for cid in sub_ids:
    rec = records[cid]
    sig = rec.get("redrob_signals") or rec.get("signals") or {}
    su = parse_date(sig.get("signup_date") or (rec.get("metadata") or {}).get("signup_date"))
    la = parse_date(sig.get("last_active_date") or (rec.get("metadata") or {}).get("last_active_date"))
    if su and la and su > la:
        other = results[cid][0]
        sig_anom.append((cid, str(su), str(la), other))
        print(f"{cid}: signup {su} > last_active {la}; hard markers: {other}")
if not sig_anom:
    print("none")

# ---------- extra organizer-style checks ----------
print("\n=== extra checks (education vs career, extreme skill durations, future dates, salary) ===")
for cid in sub_ids:
    rec = records[cid]
    prof = rec.get("profile") or {}
    career = rec.get("career_history") or []
    edu = rec.get("education") or []
    skills = rec.get("skills") or []
    notes = []
    starts = [parse_date(j.get("start_date")) for j in career if parse_date(j.get("start_date"))]
    span_mo = months_between(min(starts), TODAY) if starts else None
    # future dates anywhere
    for j in career:
        for k in ("start_date", "end_date"):
            d = parse_date(j.get(k))
            if d and d > TODAY:
                notes.append(f"FUTURE job date {k}={j.get(k)} at {j.get('company')}")
    for e in edu:
        ey = e.get("end_year"); sy = e.get("start_year")
        if ey and ey > 2026:
            notes.append(f"FUTURE edu end_year {ey}")
        if sy and ey and ey < sy:
            notes.append(f"edu end<start {sy}->{ey} (noise: reversed degree order)")
    # education vs career: graduated AFTER first job by big margin handled as known-noise (edu-to-career gaps);
    # but check: degree END year far in future relative to career, or first job BEFORE birth-ish start (start before edu start - 10y -> weird)
    # extreme skill duration vs career span (>2x and > span+60)
    if span_mo:
        for s in skills:
            dm = s.get("duration_months") or 0
            if dm > span_mo + 60 and dm > 2 * span_mo:
                notes.append(f"EXTREME skill dur {s.get('name')}={dm}mo vs career span {span_mo:.0f}mo")
    # salary corruption (min>max is noise alone; report only with another note)
    sig = rec.get("redrob_signals") or {}
    sal = sig.get("expected_salary_range_inr_lpa") or {}
    smin, smax = sal.get("min"), sal.get("max")
    if smin is not None and smax is not None and smin > smax:
        notes.append(f"salary min>max ({smin}>{smax}) [noise alone]")
    # cert year sanity (<1990?)
    for c in (rec.get("certifications") or []):
        yr = c.get("year")
        if yr is not None and yr < 1990:
            notes.append(f"cert year {yr} suspicious-old")
    interesting = [n for n in notes if not n.startswith("salary") and "noise" not in n]
    if interesting:
        print(f"{cid}: {notes}")

print("\nDone.")
