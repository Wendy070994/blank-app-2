"""
Pre-processing App
------------------
Create rows with explicit **Context** + **Statement** for
experiments in sentence/turn/post-level text-classification.

Input CSV must contain the columns:  ID, Turn, Statement
(upper/lower-case doesn’t matter; headers are normalised).

Rules implemented
1. Each *statement* row contains exactly **one sentence**.
2. Hashtags (e.g. #Deal) are kept as **stand-alone sentences**.
3. Rows made of **only punctuation** are discarded.

The sidebar lets users choose:
• statement cut   – sentence | turn | post  
• context cut     – rolling  | whole  
• speaker         – customer | salesperson
"""

from __future__ import annotations

import io
import re
from typing import List


import pandas as pd
import streamlit as st
from nltk.tokenize import sent_tokenize

# one-time download for sentence splitter
nltk.download("punkt", quiet=True)

# ───────────────────────────────────────────────────────────────────────
# Regex helpers
HASHTAG_RE = re.compile(r"(#\w+)")
PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")  # lines such as "!!!", "—", "..."

# ───────────────────────────────────────────────────────────────────────
# Sentence extraction
def _sentences_from_text(text: str) -> List[str]:
    """
    Split text into sentences while isolating hashtags.
    Returns a list filtered so that empty or punctuation-only
    fragments are removed.
    """
    if not isinstance(text, str) or text.strip() == "":
        return []

    # separate hashtags into their own sentence
    text = HASHTAG_RE.sub(r". \1 .", text.replace("\n", " ").strip())

    sentences = (s.strip() for s in sent_tokenize(text))

    return [s for s in sentences if s and not PUNCT_ONLY_RE.fullmatch(s)]


def explode_to_sentence_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per sentence."""
    rows: list[dict] = []
    for _, row in df.iterrows():
        for sent in _sentences_from_text(row["Statement"]):
            rows.append(
                {
                    "ID": row["ID"],
                    "Turn": row["Turn"],
                    "Statement": sent,
                }
            )
    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────────────────
# Context creation
def build_context_column(
    base: pd.DataFrame, context_cut: str, stmt_level: str
) -> pd.DataFrame:
    """
    Add a 'Context' column according to user selection.

    context_cut : 'rolling' | 'whole'
    stmt_level  : 'sentence' | 'turn' | 'post'
    """
    # ensure chronological order inside each post/chat
    base = base.sort_values(["ID", "Turn"]).reset_index(drop=True)

    # Whole-post context: every row gets the full post text
    if context_cut == "whole":
        whole_post_txt = base.groupby("ID")["Statement"].transform(
            lambda x: " ".join(x.tolist())
        )
        base["Context"] = whole_post_txt
        return base

    # Rolling context (everything **before** the current row in the same post)
    if stmt_level == "sentence":
        base["Context"] = (
            base.groupby("ID")
            .apply(lambda g: g["Statement"].shift().cumsum().fillna(""))
            .reset_index(level=0, drop=True)
        )

    elif stmt_level == "turn":
        # roll at the *turn* granularity, then broadcast back
        turn_text = (
            base.groupby(["ID", "Turn"])["Statement"]
            .apply(lambda x: " ".join(x))
            .groupby(level=0)
            .apply(lambda g: g.shift().cumsum().fillna(""))
        )
        base["Context"] = [turn_text.loc[idx] for idx in zip(base["ID"], base["Turn"])]

    else:  # post-level statement ⇒ rolling context is always empty
        base["Context"] = ""

    return base


# ───────────────────────────────────────────────────────────────────────
# Streamlit interface
st.set_page_config(page_title="Pre-processing App", layout="wide")
st.title("🛠️  Text Pre-processing for Classification Experiments")

with st.sidebar:
    st.header("⚙️  Options")
    stmt_level = st.selectbox("Statement cut:", ["sentence", "turn", "post"], 0)
    ctx_level = st.selectbox("Context cut:", ["rolling", "whole"], 0)
    speaker_sel = st.selectbox("Speaker:", ["customer", "salesperson"], 0)

upl = st.file_uploader(
    "📤 Upload CSV containing ID, Turn, Statement", type=["csv"], accept_multiple_files=False
)

if not upl:
    st.info("Drag a CSV here or click to browse.")
    st.stop()

# ── Load data
raw = pd.read_csv(upl)
raw.columns = [c.strip().title() for c in raw.columns]  # normalise headers

missing = {"Id", "Turn", "Statement"} - set(raw.columns)
if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

# ── Transform according to sidebar settings
if stmt_level == "sentence":
    df_stmt = explode_to_sentence_rows(raw)
elif stmt_level == "turn":
    df_stmt = raw.copy()
else:  # post
    df_stmt = (
        raw.groupby("ID", as_index=False)["Statement"]
        .apply(lambda x: " ".join(x))
        .rename(columns={0: "Statement"})
    )
    df_stmt["Turn"] = 0  # dummy (keeps ordering consistent)

df_stmt = build_context_column(df_stmt, ctx_level, stmt_level)
df_stmt["Speaker"] = speaker_sel

# ── Display + download
st.success(f"✅  Created {len(df_stmt):,} rows.")
st.dataframe(df_stmt.head(20), use_container_width=True)

buf = io.BytesIO()
df_stmt.to_csv(buf, index=False)
st.download_button(
    "⬇️  Download processed CSV",
    data=buf.getvalue(),
    file_name="processed_data.csv",
    mime="text/csv",
)
