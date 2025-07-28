"""
Pre-processing App
------------------
Creates rows with explicit Context + Statement for
sentence / turn / post-level text-classification experiments.

The uploaded CSV **must include** the columns that mean:
    ID , Turn , Statement
Capitalisation, spaces, underscores, emoji – none of that matters:
“ i_d ” or “Turn #” will still be mapped.
"""

from __future__ import annotations

import io
import re
from typing import List


import pandas as pd
import streamlit as st


# one-time download for the sentence splitter
nltk.download("punkt", quiet=True)

# ─────────────────────────────── helpers
HASHTAG_RE     = re.compile(r"(#\w+)")
PUNCT_ONLY_RE  = re.compile(r"^[\W_]+$")          # lines such as "!!!" or "—"
ALPHABETIC_RE  = re.compile(r"[^a-z]")            # drop non-letters for keying


def _canonical(col: str) -> str:
    """Return a lowercase alpha-only version of *col* (  'Turn #' → 'turn'  )."""
    return ALPHABETIC_RE.sub("", col.lower())


# ─────────────────────────────── sentence extraction
def _sentences_from_text(text: str) -> List[str]:
    """Split text, isolate hashtags, drop empty or punct-only fragments."""
    if not isinstance(text, str) or not text.strip():
        return []

    text = HASHTAG_RE.sub(r". \1 .", text.replace("\n", " ").strip())
    sentences = (s.strip() for s in sent_tokenize(text))
    return [s for s in sentences if s and not PUNCT_ONLY_RE.fullmatch(s)]


def explode_to_sentence_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per sentence (statement level)."""
    out: list[dict] = []
    for _, r in df.iterrows():
        for sent in _sentences_from_text(r["statement"]):
            out.append({"id": r["id"], "turn": r["turn"], "statement": sent})
    return pd.DataFrame(out)


# ─────────────────────────────── context creation
def build_context_column(
    base: pd.DataFrame, context_cut: str, stmt_level: str
) -> pd.DataFrame:
    """Add a *context* column as requested in the sidebar."""
    base = base.sort_values(["id", "turn"]).reset_index(drop=True)

    if context_cut == "whole":                       # full post/chat as context
        base["context"] = base.groupby("id")["statement"]\
                               .transform(" ".join)
        return base

    if stmt_level == "sentence":                     # rolling @ sentence level
        base["context"] = (
            base.groupby("id")
            .apply(lambda g: g["statement"].shift().cumsum().fillna(""))
            .reset_index(level=0, drop=True)
        )

    elif stmt_level == "turn":                       # rolling @ turn level
        turn_text = (
            base.groupby(["id", "turn"])["statement"]
            .apply(" ".join)
            .groupby(level=0)
            .apply(lambda g: g.shift().cumsum().fillna(""))
        )
        base["context"] = [turn_text.loc[idx] for idx in zip(base["id"], base["turn"])]

    else:                                            # post-level statements
        base["context"] = ""

    return base


# ─────────────────────────────── Streamlit UI
st.set_page_config(page_title="Pre-processing App", layout="wide")
st.title("🛠️  Text Pre-processing for Classification Experiments")

with st.sidebar:
    st.header("⚙️  Options")
    stmt_level  = st.selectbox("Statement cut:", ["sentence", "turn", "post"], 0)
    ctx_level   = st.selectbox("Context cut:",   ["rolling", "whole"],        0)
    speaker_sel = st.selectbox("Speaker:",       ["customer", "salesperson"], 0)

upl = st.file_uploader(
    "📤 Upload CSV containing ID, Turn, Statement", type=["csv"]
)

if not upl:
    st.info("Drag a CSV here or click to browse.")
    st.stop()


# ─────────────────────────────── load & map columns
raw = pd.read_csv(upl)

# build a mapping from “weird” header names to canonical ones
required_keys = {"id": None, "turn": None, "statement": None}

for col in raw.columns:
    key = _canonical(col)
    if key in required_keys and required_keys[key] is None:
        required_keys[key] = col

missing = [k for k, v in required_keys.items() if v is None]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

# rename to canonical for internal use
raw = raw.rename(columns={v: k for k, v in required_keys.items()})

# ensure correct dtypes
raw["turn"] = pd.to_numeric(raw["turn"], errors="coerce").fillna(0).astype(int)

# ─────────────────────────────── transform
if stmt_level == "sentence":
    df_stmt = explode_to_sentence_rows(raw)
elif stmt_level == "turn":
    df_stmt = raw.copy()
else:  # post
    df_stmt = (
        raw.groupby("id", as_index=False)["statement"]
            .apply(" ".join)
            .rename(columns={0: "statement"})
    )
    df_stmt["turn"] = 0

df_stmt = build_context_column(df_stmt, ctx_level, stmt_level)
df_stmt["speaker"] = speaker_sel

# ─────────────────────────────── display & download
st.success(f"✅  Created {len(df_stmt):,} rows.")
st.dataframe(df_stmt.head(20), use_container_width=True)

buf = io.BytesIO()
(df_stmt
 .rename(columns={"id": "ID", "turn": "Turn", "statement": "Statement",
                  "context": "Context", "speaker": "Speaker"})
).to_csv(buf, index=False)

st.download_button(
    "⬇️  Download processed CSV",
    data=buf.getvalue(),
    file_name="processed_data.csv",
    mime="text/csv",
)


