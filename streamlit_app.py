# streamlit_ig_sentence_app.py
#
# Usage:
#   streamlit run streamlit_ig_sentence_app.py
#
# Requirements:
#   pip install streamlit pandas nltk
#
# -----------------------------------------------------------------------------
import io
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

###############################################################################
# One-time NLTK setup                                                          #
###############################################################################
from nltk.tokenize import sent_tokenize

###############################################################################
# Transformation helpers (adapted unchanged from instagram_sentence_transformer) #
###############################################################################
def _normalize(text: str) -> str:
    """Straighten curly quotes and trim whitespace (light normalisation)."""
    return (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .strip()
    )

def _sentences(text: str) -> List[str]:
    """Split *text* into a list of clean sentences."""
    text = _normalize(text)

    try:
        parts = sent_tokenize(text)
    except Exception:  # fallback if Punkt struggles
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)

    sentences: List[str] = []
    for s in parts:
        s = s.strip()
        if not s:
            continue
        if s[-1] not in ".!?":
            s += "."
        sentences.append(s)

    return sentences or [text]

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Convert an Instagram-style captions table into a sentence-level table."""
    rename_map = {"shortcode": "ID", "caption": "Context"}
    df = df.rename(columns=rename_map)

    missing = {col for col in ["ID", "Context"] if col not in df.columns}
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {', '.join(sorted(missing))}"
        )

    rows: List[dict] = []
    for _, row in df.iterrows():
        for idx, s in enumerate(_sentences(str(row["Context"])), start=1):
            rows.append(
                {
                    "ID": row["ID"],
                    "Sentence ID": idx,
                    "Context": row["Context"],
                    "Statement": s,
                }
            )

    return pd.DataFrame(rows, columns=["ID", "Sentence ID", "Context", "Statement"])

###############################################################################
# Streamlit UI                                                                #
###############################################################################
st.set_page_config(page_title="IG Sentence Transformer", layout="wide")
st.title("📄→✂️ Instagram Caption Sentence Splitter")

st.markdown(
    """
Upload a **raw caption CSV** (`shortcode`, `caption` columns).  
The app splits every caption into tidy one-sentence rows that match the reference
schema—ready for downstream NLP pipelines.
"""
)

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"🚫 Could not read CSV: {e}")
        st.stop()

    st.success(f"Loaded {len(raw_df):,} rows. Preview below:")
    st.dataframe(raw_df.head(), height=250)

    with st.spinner("Transforming…"):
        try:
            out_df = transform(raw_df)
        except Exception as e:
            st.error(f"🚫 Transformation failed: {e}")
            st.stop()

    st.subheader("✅ Transformed Output (first 100 rows)")
    st.dataframe(out_df.head(100), height=400)

    # Prepare download
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download full CSV",
        data=csv_bytes,
        file_name="ig_posts_transformed.csv",
        mime="text/csv",
    )

else:
    st.info("👆 Drag & drop a CSV file above to get started.")
