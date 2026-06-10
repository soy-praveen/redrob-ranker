"""Streamlit sandbox for the Redrob ranker.

Upload a candidates JSONL sample (or use the bundled 50-candidate sample)
and get the ranked CSV. Runs the exact same pipeline as rank.py.

Deploy: streamlit run sandbox/app.py  (repo root as working dir)
"""

import json
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_features  # noqa: E402
from reasoning import generate_reasoning  # noqa: E402
from scoring import load_params, rank_pool  # noqa: E402

st.set_page_config(page_title="Redrob Ranker Sandbox", layout="wide")
st.title("Redrob Candidate Ranker — Sandbox")
st.markdown(
    "Ranks candidates for the **Senior AI Engineer (Founding Team)** JD using "
    "graded career-history evidence x JD fit x behavioral availability. "
    "No network calls, CPU-only, deterministic."
)

uploaded = st.file_uploader(
    "Upload a candidates .jsonl sample (one JSON object per line, schema per "
    "candidate_schema.json). Leave empty to use the bundled 50-candidate sample.",
    type=["jsonl", "json"],
)

top_n = st.slider("Shortlist size", 5, 100, 20)

if st.button("Rank candidates", type="primary"):
    t0 = time.time()
    if uploaded is not None:
        tmp = ROOT / "sandbox" / "_uploaded.jsonl"
        tmp.write_bytes(uploaded.getvalue())
        src_path = tmp
    else:
        src_path = ROOT / "sandbox" / "sample_50.jsonl"

    with st.spinner("Extracting features and scoring..."):
        df = build_features(str(src_path))
        params = load_params(str(ROOT / "config" / "params.json"))
        n = min(top_n, len(df))
        scored, ranked = rank_pool(df, params, top_n=n)
        grades = dict(params["para_grades"])
        ranked["_grades_by_prefix"] = [grades] * len(ranked)
        ranked["_evidence_tags"] = [params["evidence_tags"]] * len(ranked)
        ranked["reasoning"] = [
            generate_reasoning(row, row["rank"]) for _, row in ranked.iterrows()
        ]

    st.success(
        f"Ranked {len(df)} candidates in {time.time() - t0:.1f}s — "
        f"{(scored['exclusion'] != '').sum()} excluded "
        f"(honeypots / keyword stuffers / clones / non-technical)."
    )

    out = ranked[["candidate_id", "rank", "score", "reasoning"]]
    st.dataframe(out, use_container_width=True, hide_index=True)
    st.download_button(
        "Download submission CSV",
        out.to_csv(index=False).encode("utf-8"),
        file_name="submission_sample.csv",
        mime="text/csv",
    )

    with st.expander("Why were candidates excluded?"):
        excl = scored[scored["exclusion"] != ""][
            ["candidate_id", "current_title", "exclusion"]
        ]
        st.dataframe(excl, use_container_width=True, hide_index=True)

    with st.expander("Score components (top of list)"):
        comp = ranked[
            ["rank", "candidate_id", "current_title", "best_grade",
             "score_C", "score_F", "score_A", "score_P", "final"]
        ].round(2)
        st.dataframe(comp, use_container_width=True, hide_index=True)
