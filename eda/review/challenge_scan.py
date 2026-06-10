# -*- coding: utf-8 -*-
"""Adversarial rank-quality scan. Recomputes EVERYTHING from raw JSONL.
Outputs: per-candidate feature table for all template/grade>=3 holders +
the submission 100, with honeypot markers, signals, location, grades.
"""
import json, re, csv, sys
from datetime import date

TODAY = date(2026, 6, 10)
ROOT = "c:/Users/yosoy/OneDrive/Desktop/Github/Hackathons"
JSONL = (ROOT + "/hack2skill/The Data & AI Challenge/"
         "[PUB] India_runs_data_and_ai_challenge/"
         "India_runs_data_and_ai_challenge/candidates.jsonl")
SUB = ROOT + "/redrob-ranker/submission_draft.csv"
CONS = ROOT + "/redrob-ranker/eda/round2/paragraph_grades_consensus.json"
OUT = ROOT + "/redrob-ranker/eda/review/scan_table.csv"
RAWOUT = ROOT + "/redrob-ranker/eda/review/raw_of_interest.jsonl"

ELITE = "hands-on experience building production ML systems, with a focus on search, retrieval, and ranking"
PLAIN = "building systems that connect users with relevant information at scale"

cons = json.load(open(CONS, encoding="utf-8"))
PREF = {c["text_prefix_60"][:58]: (c["pid"], c["grade"]) for c in cons}

sub_rows = list(csv.DictReader(open(SUB, encoding="utf-8")))
sub_rank = {r["candidate_id"]: int(r["rank"]) for r in sub_rows}

YOE_PAT = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)(?:'| of)?\s*(?:hands-on\s+)?experience")
SIX_FIRMS = ("tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini")

def parse_date(s):
    if not s: return None
    y, m, d = s.split("-"); return date(int(y), int(m), int(d))

def months_between(a, b): return (b - a).days / 30.4375

def markers(r):
    out = []
    prof = r.get("profile", {}); jobs = r.get("career_history") or []
    if any(s.get("proficiency") == "expert" and s.get("duration_months") == 0
           for s in (r.get("skills") or [])): out.append("A")
    for j in jobs:
        sd, ed, dur = parse_date(j.get("start_date")), parse_date(j.get("end_date")), j.get("duration_months")
        if sd and dur is not None:
            eff = TODAY if (j.get("is_current") or ed is None) else min(ed, TODAY)
            if abs(dur - months_between(sd, eff)) > 3: out.append("B"); break
    starts = [parse_date(j.get("start_date")) for j in jobs if j.get("start_date")]
    ends = [TODAY if (j.get("is_current") or not j.get("end_date"))
            else min(parse_date(j["end_date"]), TODAY) for j in jobs]
    yoe = prof.get("years_of_experience")
    if starts and yoe is not None:
        if yoe * 12 > months_between(min(starts), max(ends)) + 24: out.append("C")
    m = YOE_PAT.search(prof.get("summary") or "")
    if m and yoe is not None and abs(float(m.group(1)) - yoe) > 1.5: out.append("D")
    if any((c.get("year") or 0) > 2026 for c in (r.get("certifications") or [])): out.append("E")
    return "".join(out)

def grade_info(r):
    pids, best = [], -1
    for j in (r.get("career_history") or []):
        d = (j.get("description") or "").strip()
        if not d: continue
        hit = PREF.get(d[:58])
        if hit is None:
            # fallback: scan all prefixes as substrings (paranoia)
            for k, v in PREF.items():
                if k in d: hit = v; break
        if hit: pids.append(hit)
        else: pids.append(("P??", -9))
    if pids: best = max(g for _, g in pids)
    return best, "|".join(f"{p}:{g}" for p, g in pids)

def main():
    rows = []
    raw_keep = open(RAWOUT, "w", encoding="utf-8")
    n = 0
    stuffer_phrase = "online courses on RAG and vector databases"
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            n += 1
            r = json.loads(line)
            cid = r["candidate_id"]
            prof = r["profile"]; sig = r.get("redrob_signals", {})
            summ = prof.get("summary") or ""
            is_elite = ELITE in summ
            is_plain = PLAIN in summ
            in_sub = cid in sub_rank
            best, pidstr = grade_info(r)
            if not (is_elite or is_plain or in_sub or best >= 3):
                continue
            la = parse_date(sig.get("last_active_date"))
            days_active = (TODAY - la).days if la else 9999
            resp = sig.get("recruiter_response_rate")
            jobs = r.get("career_history") or []
            comps = [(j.get("company") or "").lower() for j in jobs]
            consult_only = bool(comps) and all(any(fx in c for fx in SIX_FIRMS) for c in comps)
            ended = [j for j in jobs if j.get("end_date")]
            short_ended = sum(1 for j in ended if (j.get("duration_months") or 99) < 20)
            durs = [j.get("duration_months") for j in jobs if j.get("duration_months") is not None]
            avg_ten = sum(durs)/len(durs) if durs else None
            mk = markers(r)
            rows.append(dict(
                candidate_id=cid, rank=sub_rank.get(cid, ""),
                elite=int(is_elite), plain=int(is_plain),
                best_grade=best, pids=pidstr, markers=mk,
                title=prof.get("current_title"), company=prof.get("current_company"),
                yoe=prof.get("years_of_experience"),
                city=prof.get("location"), country=prof.get("country"),
                reloc=sig.get("willing_to_relocate"),
                resp=resp, days_active=days_active,
                notice=sig.get("notice_period_days"),
                open_to_work=sig.get("open_to_work_flag"),
                stuffer=int(stuffer_phrase in summ),
                n_jobs=len(jobs), short_ended=short_ended,
                avg_tenure=round(avg_ten,1) if avg_ten is not None else "",
                consult_only=int(consult_only),
                ghost=int((resp is not None and resp < 0.2) and days_active > 100),
            ))
            if is_elite or is_plain or in_sub:
                raw_keep.write(line)
    raw_keep.close()
    cols = list(rows[0].keys())
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"scanned {n} records; kept {len(rows)} rows -> {OUT}")
    print(f"elite holders: {sum(r['elite'] for r in rows)}; plain: {sum(r['plain'] for r in rows)}")
    print(f"submission ids found in raw: {sum(1 for r in rows if r['rank'] != '')} / 100")
    print(f"grade>=4 holders: {sum(1 for r in rows if r['best_grade'] >= 4)}")
    print(f"grade==5: {sum(1 for r in rows if r['best_grade'] == 5)}")
    unk = [r for r in rows if ':-9' in r['pids']]
    print(f"rows with unmatched paragraph: {len(unk)}")
    for r in unk[:10]: print("  UNMATCHED", r['candidate_id'], r['pids'])

if __name__ == "__main__":
    main()
