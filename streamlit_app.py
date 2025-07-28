"""
📝 Text Transformation App
==========================

Split long captions (or any free-text) into individual sentences.

Workflow
1. Upload a CSV that contains an **ID** column (post identifier) and a
   **Context** column (text/caption).
2. Pick those two columns from the dropdowns.
3. Choose whether to pull out hashtags as standalone sentences.
4. Click **Process** – preview appears and you can download the result.

Output columns
• ID          – value from your chosen ID column  
• Sentence ID – sequential number within each original record  
• Context     – original full caption/text row  
• Statement   – one sentence extracted from *Context*
"""

from __future__ import annotations

import io
import re
from typing import List


import pandas as pd
import streamlit as st


# one-time download for sentence splitter
nltk.download("punkt", quiet=True)

# ------------------------------------------------------------------ helpers
PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")          # e.g. '!!!'
HASHTAG_RE    = re.compile(r"(#\w+)")            # captures #Sale, #2024 etc.


def split_into_sentences(
    text: str,
    isolate_hashtags: bool = True,
) -> List[str]:
    """
    Break *text* into sentences.

    • If *isolate_hashtags* is True, hashtags become their own sentences.
    • Sentences that are only punctuation are discarded.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    if isolate_hashtags:
        text = HASHTAG_RE.sub(r". \1 .", text)

    raw_sents = (s.strip() for s in sent_tokenize(text.replace("\n", " ")))

    return [s for s in raw_sents if s and not PUNCT_ONLY_RE.fullmatch(s)]


# ------------------------------------------------------------------ page config
st.set_page_config(page_title="📝 Text Transformation App", layout="wide")
st.title("📝 Text Transformation App")

with st.expander("ℹ️  How to Use", expanded=True):
    st.markdown(
        """
        1. **Upload** your CSV containing Instagram (or other) text.
        2. **Select** the *ID* column (unique post/chat ID) and the *Context*
           column (caption/text to split).
        3. **Options**  
           • *Hashtags as separate sentences* – if ticked, `#Tags` become
             one-sentence rows.  
        4. Click **Process** to preview and **Download** the transformed file.
        """
    )

# ------------------------------------------------------------------ upload
csv_file = st.file_uploader("📤 Upload CSV", type=["csv"])

if csv_file is None:
    st.stop()

df_raw = pd.read_csv(csv_file)

if df_raw.empty:
    st.error("Uploaded file is empty.")
    st.stop()

# ------------------------------------------------------------------ column pickers
st.sidebar.header("⚙️  Configuration")

id_col = st.sidebar.selectbox("Select ID column (unique identifier)", df_raw.columns)
ctx_col = st.sidebar.selectbox("Select Context column (text to split)", df_raw.columns)

isolate_hash = st.sidebar.checkbox(
    "Treat hashtags as separate sentences", value=True
)

if not id_col or not ctx_col:
    st.info("Choose both columns to proceed.")
    st.stop()

# ------------------------------------------------------------------ processing
if st.sidebar.button("Process"):
    st.subheader("Preview")

    rows = []
    for _, row in df_raw.iterrows():
        sentences = split_into_sentences(row[ctx_col], isolate_hash)
        for sid, sent in enumerate(sentences, 1):
            rows.append(
                {
                    "ID": row[id_col],
                    "Sentence ID": sid,
                    "Context": row[ctx_col],
                    "Statement": sent,
                }
            )

    if not rows:
        st.warning("No sentences extracted – check your settings.")
        st.stop()

    df_out = pd.DataFrame(rows)

    st.success(f"Generated {len(df_out):,} sentence rows.")
    st.dataframe(df_out.head(20), use_container_width=True)

    # download
    buffer = io.BytesIO()
    df_out.to_csv(buffer, index=False)
    st.download_button(
        "⬇️  Download processed CSV",
        data=buffer.getvalue(),
        file_name="processed_data.csv",
        mime="text/csv",
    )
else:
    st.info("Adjust options, then click **Process** in the sidebar.")


