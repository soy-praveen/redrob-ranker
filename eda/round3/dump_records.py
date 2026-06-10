import json, csv, sys
from datetime import date
RAW = r"C:\Users\yosoy\OneDrive\Desktop\Github\Hackathons\hack2skill\The Data & AI Challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
SUB = r"C:\Users\yosoy\OneDrive\Desktop\Github\Hackathons\redrob-ranker\submission_draft.csv"
TODAY = date(2026, 6, 10)
lo, hi = int(sys.argv[1]), int(sys.argv[2])
ids_arg = sys.argv[3].split(",") if len(sys.argv) > 3 else None

rows = list(csv.DictReader(open(SUB, encoding="utf-8")))
if ids_arg:
    want = {r["candidate_id"]: int(r["rank"]) for r in rows if r["candidate_id"] in ids_arg}
else:
    want = {r["candidate_id"]: int(r["rank"]) for r in rows if lo <= int(r["rank"]) <= hi}

def pd(s):
    if not s: return None
    y, m, d = s.split("-"); return date(int(y), int(m), int(d))
def mb(d1, d2):
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) + (d2.day - d1.day) / 30.44

recs = {}
with open(RAW, encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        if rec["candidate_id"] in want:
            recs[rec["candidate_id"]] = rec
            if len(recs) == len(want): break

for cid, rank in sorted(want.items(), key=lambda kv: kv[1]):
    rec = recs[cid]; p = rec["profile"]; sig = rec["redrob_signals"]
    print("=" * 110)
    print(f"RANK {rank}  {cid}  yoe={p['years_of_experience']}  {p['current_title']} @ {p['current_company']}  loc={p['location']},{p['country']}")
    print(f"HEADLINE: {p['headline']}")
    print(f"SUMMARY: {p['summary']}")
    starts = [pd(j["start_date"]) for j in rec["career_history"] if pd(j["start_date"])]
    span = mb(min(starts), TODAY) if starts else 0
    print(f"CAREER (span {span:.1f}mo = {span/12:.1f}y vs yoe {p['years_of_experience']} -> yoe*12={p['years_of_experience']*12:.0f}):")
    for j in rec["career_history"]:
        sd, ed = pd(j["start_date"]), pd(j["end_date"]) or TODAY
        diff = j["duration_months"] - mb(sd, ed) if sd else float("nan")
        print(f"  {j['start_date']} .. {j['end_date'] or 'now'}  dur={j['duration_months']}mo (delta {diff:+.1f})  {j['title']} @ {j['company']} [{j['industry']}]")
        print(f"    DESC: {j['description']}")
    for e in rec["education"]:
        print(f"EDU: {e['start_year']}-{e['end_year']} {e['degree']} {e['field_of_study']} @ {e['institution']} ({e['tier']}, grade {e.get('grade')})")
    sk = sorted(rec["skills"], key=lambda s: -(s["duration_months"] or 0))
    print("SKILLS: " + "; ".join(f"{s['name']}({s['proficiency']},{s['duration_months']}mo,e{s['endorsements']})" for s in sk))
    print("CERTS: " + (json.dumps(rec["certifications"]) if rec["certifications"] else "none"))
    sal = sig["expected_salary_range_inr_lpa"]
    print(f"SIGNALS: signup={sig['signup_date']} last_active={sig['last_active_date']} resp={sig['recruiter_response_rate']} notice={sig['notice_period_days']}d sal={sal['min']}-{sal['max']} otw={sig['open_to_work_flag']} compl={sig['profile_completeness_score']}")
