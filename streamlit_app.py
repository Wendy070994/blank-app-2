# streamlit_app.py
"""
📝 Text Transformation App – Streamlit
--------------------------------------

Upload any CSV that has:
• one column that uniquely identifies each record (ID)
• one column that contains free-text (Context)

The app will split Context into sentence-level rows and, using a
keyword dictionary, label each sentence with a simple category.

Output columns
• ID          – value from the chosen ID column
• Sentence ID – sequential number inside each original record
• Context     – original full text
• Statement   – extracted sentence
• Category    – first matching category from the keyword dictionary
"""

from io import StringIO
import json
import re
from typing import List

import pandas as pd
import streamlit as st


# ─────────────────────────────  Page set-up  ─────────────────────────────
st.set_page_config(
    page_title="Text Transformation App",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject a bit of CSS for nicer spacing / centring
st.markdown(
    """
<style>
    .block-container h1:first-child{
        text-align:center;font-size:3rem;font-weight:800;margin-bottom:1.2rem}
    .block-container{padding-top:1.2rem}
    section[data-testid="stSidebar"] h2{
        font-size:1.05rem;margin-bottom:.3rem}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📝 Text Transformation App")

# ─────────────────────────────  Default keyword dictionary  ─────────────────────────────
DEFAULT_DICT = {
    "Fashion": ["style", "fashion", "wardrobe", "clothing", "outfit"],
    "Food": ["delicious", "food", "dinner", "lunch", "restaurant"],
    "Travel": ["travel", "trip", "vacation", "explore", "journey"],
    "Fitness": ["workout", "fitness", "exercise", "gym", "training"],
}

# ─────────────────────────────  Helper functions  ─────────────────────────────
PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")   # rows like "!!!"
HASHTAG_RE = re.compile(r"(#\\w+)")        # capture #Tags


def split_sentences(text: str, isolate_hashtags: bool) -> List[str]:
    """
    Break *text* into individual sentences.

    If *isolate_hashtags* is True, hashtags become their own sentences.
    Sentences containing only punctuation are discarded.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    if isolate_hashtags:
        text = HASHTAG_RE.sub(r". \\1 .", text)

    parts = re.split(r"(?<=[.!?])\\s+|(?=#[^\\s]+)", text.replace("\\n", " "))
    clean = [p.strip() for p in parts if p.strip()]

    return [c for c in clean if not PUNCT_ONLY_RE.fullmatch(c)]


def classify(sentence: str, kw_dict: dict) -> str:
    """Return the first category whose keywords appear in *sentence* (case-insensitive)."""
    low = sentence.lower()
    for cat, kws in kw_dict.items():
        if any(k.lower() in low for k in kws):
            return cat
    return "Uncategorized"


def transform_df(
    df: pd.DataFrame,
    id_col: str,
    ctx_col: str,
    kw_dict: dict,
    isolate_hashtags: bool,
) -> pd.DataFrame:
    """Core transformation: split, classify, tidy."""
    rows = []
    for _, r in df.iterrows():
        sents = split_sentences(r[ctx_col], isolate_hashtags)
        for sid, sent in enumerate(sents, 1):
            rows.append(
                {
                    "ID": r[id_col],
                    "Sentence ID": sid,
                    "Context": r[ctx_col],
                    "Statement": sent,
                    "Category": classify(sent, kw_dict),
                }
            )
    return pd.DataFrame(rows)


# ─────────────────────────────  Sidebar  ─────────────────────────────
## 1️⃣  Upload
st.sidebar.header("1️⃣  Upload CSV")
file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if file is None:
    st.info("👈  Upload a CSV from the sidebar to begin.")
    st.stop()

df_input = pd.read_csv(file)

## 2️⃣  Column selection
st.sidebar.header("2️⃣  Select columns")
id_column = st.sidebar.selectbox("ID column (unique key)", df_input.columns)
ctx_column = st.sidebar.selectbox("Context column (text)", df_input.columns)

## 3️⃣  Options
st.sidebar.header("3️⃣  Options")
isolate_hashtags = st.sidebar.checkbox("Treat hashtags as separate sentences", True)

## 4️⃣  Keyword dictionary
st.sidebar.header("4️⃣  Modify keyword dictionary")
dict_text = st.sidebar.text_area(
    "Dictionary (JSON – edit as needed)",
    value=json.dumps(DEFAULT_DICT, indent=2),
    height=280,
)

try:
    KEYWORD_DICT = json.loads(dict_text)
    if not isinstance(KEYWORD_DICT, dict):
        raise ValueError
except Exception:
    st.sidebar.error("❌ Invalid JSON. Falling back to default dictionary.")
    KEYWORD_DICT = DEFAULT_DICT

# ─────────────────────────────  Instructions (always visible)  ─────────────────────────────
st.markdown(
    """
### How to Use
1. **Upload** your CSV file in the sidebar.  
2. **Select** the *ID* and *Context* columns.  
3. **Configure options** – Choose whether hashtags become separate sentences.  
4. **Edit** the keyword dictionary if you wish.  
5. Click **Transform** to process your data.  
6. **Download** the results as a new CSV.

### Output Format
| Column       | Description                                                |
|--------------|------------------------------------------------------------|
| `ID`         | The identifier from your selected ID column                |
| `Sentence ID`| Sequential number for each sentence inside a record        |
| `Context`    | The original text from your Context column                 |
| `Statement`  | Individual sentence extracted from the context             |
| `Category`   | First matching category based on the keyword dictionary    |
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────  Transform button  ─────────────────────────────
if st.sidebar.button("⚙️  Transform"):
    with st.spinner("Processing…"):
        output_df = transform_df(
            df_input, id_column, ctx_column, KEYWORD_DICT, isolate_hashtags
        )

    st.success(f"Done! Extracted {len(output_df):,} sentences.")
    st.dataframe(output_df.head(25), use_container_width=True)

    buff = StringIO()
    output_df.to_csv(buff, index=False)
    st.download_button(
        "💾  Download CSV",
        data=buff.getvalue(),
        file_name="transformed_text.csv",
        mime="text/csv",
    )

