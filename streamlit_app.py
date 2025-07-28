# streamlit_app.py — drop expander, always visible instructions
"""
📝 Text Transformation App – Streamlit
-------------------------------------
Changes in this patch:
 
* **Removed** the collapsible expander. The How‑to section now shows by default.
* No other behaviour changed.
"""
 
from io import StringIO
import json
import re
 
import pandas as pd
import streamlit as st
 
# ──────────────────────────────  Page set‑up  ──────────────────────────────
st.set_page_config(
    page_title="Text Transformation App",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# Inject CSS for centred hero + compact padding
st.markdown(
    """
<style>
        /* ─── Hero title ─── */
        .block-container h1:first-child {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 1.25rem;
        }
        /* ─── Compact top padding ─── */
        .block-container { padding-top: 1.2rem; }
        /* ─── Sidebar header size ─── */
        section[data-testid="stSidebar"] h2 {
            font-size: 1.05rem; margin-bottom: .3rem;
        }
</style>
    """,
    unsafe_allow_html=True,
)
 
st.title("📝 Text Transformation App")
 
# ──────────────────────────────  Default dictionary  ──────────────────────────────
DEFAULT_DICT = {
    "Fashion": ["style", "fashion", "wardrobe", "clothing", "outfit"],
    "Food": ["delicious", "food", "dinner", "lunch", "restaurant"],
    "Travel": ["travel", "trip", "vacation", "explore", "journey"],
    "Fitness": ["workout", "fitness", "exercise", "gym", "training"],
}
 
# ──────────────────────────────  Sidebar  ──────────────────────────────
## 1️⃣  Upload
st.sidebar.header("1️⃣  Upload your CSV")
uploaded_file = st.sidebar.file_uploader(
    "Choose an Instagram CSV file (≤200 MB)",
    type=["csv"],
    accept_multiple_files=False,
)
 
st.sidebar.markdown("---")
 
## 2️⃣  Dictionary
st.sidebar.header("2️⃣  Modify keyword dictionary")
 
dict_input = st.sidebar.text_area(
    "Dictionary (JSON)",
    value=json.dumps(DEFAULT_DICT, indent=2),
    height=280,
)
 
try:
    keyword_dict = json.loads(dict_input)
    if not isinstance(keyword_dict, dict):
        raise ValueError("Dictionary root must be a JSON object (key → list).")
except (json.JSONDecodeError, ValueError) as e:
    st.sidebar.error(f"❌ {e}\nUsing default dictionary instead.")
    keyword_dict = DEFAULT_DICT
 
st.sidebar.markdown(
    """<small>🔧 Edit the JSON above to add/delete categories & keywords.</small>""",
    unsafe_allow_html=True,
)
 
# ──────────────────────────────  Helper functions  ──────────────────────────────
 
def classify_sentence(text: str, kw_dict: dict) -> str:
    """Return first category whose keywords appear in *text* (case‑insensitive)."""
    text_lower = text.lower()
    for cat, kws in kw_dict.items():
        if any(k.lower() in text_lower for k in kws):
            return cat
    return "Uncategorized"
 
 
def process_dataframe(df: pd.DataFrame, kw_dict: dict) -> pd.DataFrame:
    """Split captions into sentences/hashtags ➜ classify ➜ tidy DataFrame."""
    df = df.rename(columns={"shortcode": "ID", "caption": "Context"})
    rows = []
    for _, row in df.iterrows():
        # split on sentence terminators **or** look‑ahead for a hashtag
        sentences = re.split(r"(?<=[.!?])\s+|(?=#[^\s]+)", str(row["Context"]))
        for i, s in enumerate(sentences, start=1):
            clean = re.sub(r"\s+", " ", s.strip())
            if clean and not re.fullmatch(r"[.!?]+", clean):
                rows.append(
                    {
                        "ID": row["ID"],
                        "Context": row["Context"],
                        "Sentence ID": i,
                        "Statement": clean,
                        "Category": classify_sentence(clean, keyword_dict),
                    }
                )
    return pd.DataFrame(rows)
 
# ──────────────────────────────  How to Use  ──────────────────────────────
 
st.markdown(
    """
### How to Use
1. **Upload your CSV file** using the file uploader above  
2. **Select ID Column** – Choose the column that uniquely identifies each record  
3. **Select Context Column** – Choose the column containing the text to be transformed  
4. **Configure options** – Choose whether to include hashtags as separate sentences  
5. **Click _Transform_** – Process your data into sentence‑level format  
6. **Download results** – Get your transformed data as a CSV file  
 
### Output Format
The transformed data will have the following columns:
 
- **ID**: The identifier from your selected ID column  
- **Sentence ID**: Sequential number for each sentence within a record  
- **Context**: The original text from your Context column  
- **Statement**: Individual sentences extracted from the context  
    """
)
 
# ──────────────────────────────  Main logic  ──────────────────────────────
if uploaded_file is None:
    st.info("👈  Start by uploading a CSV file from the sidebar.")
    st.stop()
 
raw_df = pd.read_csv(uploaded_file)
 
if not {"shortcode", "caption"}.issubset(raw_df.columns):
    st.error("CSV must contain `shortcode` and `caption` columns.")
    st.stop()
 
if st.sidebar.button("⚙️  Transform"):
    with st.spinner("Processing …"):
        final_df = process_dataframe(raw_df, keyword_dict)
 
    st.success("Processing complete!")
    st.subheader("Preview of processed data")
    st.dataframe(final_df, use_container_width=True)
 
    buff = StringIO()
    final_df.to_csv(buff, index=False)
    st.download_button(
        "💾  Download CSV",
        data=buff.getvalue(),
        mime="text/csv",
        file_name="transformed_text.csv",
    )
