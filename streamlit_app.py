"""
Pre-processing App
------------------
Creates rows with explicit Context + Statement for experiments in
sentence/turn/post-level text classification.

Required columns in the uploaded CSV (case-insensitive):
    id , turn , statement
"""

from __future__ import annotations

import io
import re
from typing import List

import nltk
import pandas as pd
import streamlit as st
from nltk.tokenize import sent_tokenize

# one-time download for the sentence splitter
nltk.download("punkt", quiet=True)

# ────────────────────────────────────────────────────────────
# Regex helpers
HASHTAG_RE = re.compile(r"(#\w+)")
PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")

# ────────────────────────────────────────────────────────────
# Sentence extraction
def _sentences_from_text(text: str) -> List[str]:
    """Split text into sentences, isolate hashtags, drop punct-only rows."""
    if not isinstance(text, str) or not text.strip():
        return []

    # make hashtags stand-alone sentences
    text = HASHTAG_RE.sub(r". \1 .", text.replace("\n", " ").strip())

    sentences = (s.strip() for s in sent_tokenize(text))
    return [s for s in sentences if s and not PUNCT_ONLY_RE.fullmatch(s)]


def explode_to_sentence_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per sentence (statement level)."""
    rows: list[dict] = []
    for _, row in df.iterrows():
        for sent in _sentences_from_text(row["statement"]):
            rows.append(
                {"id": row["id"], "turn": row["turn"], "statement": sent}
            )
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# Context creation
def build_context_column(
    base: pd.DataFrame, context_cut: str, stmt_level: str
) -> pd.DataFrame:
    """Add a `context` column according to user selection."""
    base = base.sort_values(["id", "turn"]).reset_index(drop=True)

    if context_cut == "whole":  # full post/chat as context
        base["context"] = base.groupby("id")["statement"].transform(
            lambda x: " ".join(x.tolist())
        )
        return base

    # rolling (everything before current row)
    if stmt_level == "sentence":
        base["context"] = (
            base.groupby("id")
            .apply(lambda g: g["statement"].shift().cumsum().fillna(""))
            .reset_index(level=0, drop=True)
        )
    elif stmt_level == "turn":
        turn_text = (
            base.groupby(["id", "turn"])["statement"]
            .apply(lambda x: " ".join(x))
            .groupby(level=0)
            .apply(lambda g: g.shift().cumsum().fillna(""))
        )
        base["context"] = [turn_text.loc[idx] for idx in zip(base["id"], base["turn"])]
    else:                      # post-level statements
        base["context"] = ""

    return base


# ────────────────────────────────────────────────────────────
# Streamlit UI
st.set_page_config(page_title="Pre-processing App", layout="wide")
st.title("🛠️  Text Pre-processing for Classification Experiments")

with st.sidebar:
    st.header("⚙️  Options")
    stmt_level = st.selectbox("Statement cut:", ["sentence", "turn", "post"], 0)
    ctx_level = st.selectbox("Context cut:", ["rolling", "whole"], 0)
    speaker_sel = st.selectbox("Speaker:", ["customer", "salesperson"], 0)

upl = st.file_uploader(
    "📤 Upload CSV containing ID, Turn, Statement", type=["csv"]
)

if not upl:
    st.info("Drag a CSV here or click to browse.")
    st.stop()

# ── Load & normalise column names
raw = pd.read_csv(upl)
raw.columns = [c.strip().lower() for c in raw.columns]      # <- key change

expected = {"id", "turn", "statement"}
missing = expected - set(raw.columns)
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

# ── Transform
if stmt_level == "sentence":
    df_stmt = explode_to_sentence_rows(raw)
elif stmt_level == "turn":
    df_stmt = raw.copy()
else:  # post
    df_stmt = (
        raw.groupby("id", as_index=False)["statement"]
        .apply(lambda x: " ".join(x))
        .rename(columns={0: "statement"})
    )
    df_stmt["turn"] = 0

df_stmt = build_context_column(df_stmt, ctx_level, stmt_level)
df_stmt["speaker"] = speaker_sel

# ── Display & download
st.success(f"✅  Created {len(df_stmt):,} rows.")
st.dataframe(df_stmt.head(20), use_container_width=True)

buf = io.BytesIO()
df_stmt.rename(
    columns={"id": "ID", "turn": "Turn", "statement": "Statement",
             "context": "Context", "speaker": "Speaker"}
).to_csv(buf, index=False)

st.download_button(
    "⬇️  Download processed CSV",
    data=buf.getvalue(),
    file_name="processed_data.csv",
    mime="text/csv",
)
