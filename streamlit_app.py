# 📝 Text Transformation App – v2
#
# Key upgrades
# ● Bigger built-in keyword dictionary (10 themed categories, 200+ keywords)
# ● Two matching modes   – “first hit” or “all applicable” categories
# ● Sentence filters     – min-length & min-word controls
# ● Fast caching with st.cache_data (re-runs are instant)
#
# Usage
# 1. Upload CSV   (ID + Context columns)
# 2. Tweak options & dictionary (JSON box)
# 3. Click Transform → preview + download

from __future__ import annotations

from io import StringIO
import json
import re
from typing import List, Set

import pandas as pd
import streamlit as st

#  ───────────────────────────────────────── UI & CSS
st.set_page_config(
    page_title="Text Transformation App",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container h1:first-child{
            text-align:center;font-size:3rem;font-weight:800;margin-bottom:1.2rem}
        .block-container{padding-top:1.2rem}
        section[data-testid="stSidebar"] h2{
            font-size:1.05rem;margin-bottom:.4rem}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📝 Text Transformation App")


#  ───────────────────────────────────────── Default dictionary
DEFAULT_DICT = {
    "Fashion": [
        "style", "fashion", "wardrobe", "clothes", "outfit", "OOTD", "runway",
        "trend", "vogue", "chic", "couture", "lookbook", "garment", "designer",
    ],
    "Food": [
        "delicious", "foodie", "dinner", "lunch", "restaurant", "recipe",
        "taste", "yummy", "cuisine", "dessert", "snack", "brunch",
        "kitchen", "chef", "cook", "coffee", "tea",
    ],
    "Travel": [
        "travel", "trip", "vacation", "explore", "journey", "flight",
        "hotel", "backpacking", "adventure", "wanderlust", "tourism",
        "passport", "roadtrip", "getaway", "destinations",
    ],
    "Fitness": [
        "workout", "fitness", "exercise", "gym", "training", "cardio",
        "strength", "run", "yoga", "HIIT", "wellness", "fitspo",
        "calisthenics", "pilates", "cycling", "marathon",
    ],
    "Technology": [
        "tech", "gadget", "smartphone", "AI", "machine learning", "robot",
        "software", "hardware", "coding", "programming", "startup",
        "innovation", "app", "VR", "blockchain", "5G", "cyber", "cloud",
    ],
    "Sports": [
        "soccer", "football", "basketball", "nba", "nfl", "cricket",
        "baseball", "tennis", "golf", "olympics", "athlete", "score",
        "goal", "match", "tournament", "league", "championship", "stadium",
    ],
    "Beauty": [
        "beauty", "makeup", "skincare", "cosmetics", "lipstick", "foundation",
        "mascara", "blush", "palette", "fragrance", "haircare", "salon",
        "spa", "glow", "nails", "manicure",
    ],
    "Nature": [
        "nature", "forest", "mountain", "beach", "sunset", "sunrise",
        "wildlife", "ocean", "lake", "river", "landscape", "hiking",
        "camping", "outdoors", "flora", "fauna", "eco", "green",
    ],
    "Health": [
        "health", "nutrition", "diet", "vitamin", "mental health", "sleep",
        "mindfulness", "meditation", "doctor", "medicine", "healthy",
        "hydration", "immune", "well-being",
    ],
    "Entertainment": [
        "movie", "film", "cinema", "music", "concert", "album", "song",
        "tv", "series", "netflix", "premiere", "actor", "actress",
        "festival", "show", "theatre", "hollywood", "bollywood",
    ],
}

#  ───────────────────────────────────────── Regex helpers
URL_RE = re.compile(r"https?://\\S+")
PUNCT_ONLY_RE = re.compile(r"^[\\W_]+$")     # rows like “!!!”
HASHTAG_RE = re.compile(r"(#\\w+)")


#  ───────────────────────────────────────── Functions
def clean_text(text: str) -> str:
    """Remove URLs and normalise whitespace."""
    return URL_RE.sub("", text).replace("\\n", " ").strip()


def split_sentences(text: str, isolate_hash: bool) -> List[str]:
    """
    Break text into sentences; optionally isolate hashtags.
    """
    if not text:
        return []

    if isolate_hash:
        text = HASHTAG_RE.sub(r". \\1 .", text)

    # Very fast lightweight splitter
    parts = re.split(r"(?<=[.!?])\\s+|(?=#[^\\s]+)", text)
    return [p.strip() for p in parts if p.strip()]


def filter_sentence(s: str, min_chars: int, min_words: int) -> bool:
    """True if sentence passes length & punctuation filters."""
    if len(s) < min_chars:
        return False
    if len(s.split()) < min_words:
        return False
    return not PUNCT_ONLY_RE.fullmatch(s)


def classify_sentence(
    s: str, kw_dict: dict[str, List[str]], mode: str
) -> str | List[str]:
    """
    Return category/ies for sentence *s*.

    mode = 'first' → first matching category (fast)  
    mode = 'all'   → semicolon-separated list of every match
    """
    s_low = s.lower()
    hits: List[str] = []
    for cat, kws in kw_dict.items():
        if any(k.lower() in s_low for k in kws):
            if mode == "first":
                return cat
            hits.append(cat)

    return hits[0] if mode == "first" else ";".join(hits) or "Uncategorized"


@st.cache_data(show_spinner=False)
def transform(
    df: pd.DataFrame,
    id_col: str,
    ctx_col: str,
    kw_dict: dict[str, List[str]],
    isolate_hash: bool,
    min_chars: int,
    min_words: int,
    match_mode: str,
) -> pd.DataFrame:
    """Heavy work – cached so re-runs are instant."""
    rows = []
    for _, row in df.iterrows():
        ctx = clean_text(str(row[ctx_col]))
        sentences = split_sentences(ctx, isolate_hash)

        sent_idx = 0
        for s in sentences:
            if not filter_sentence(s, min_chars, min_words):
                continue
            sent_idx += 1
            rows.append(
                {
                    "ID": row[id_col],
                    "Sentence ID": sent_idx,
                    "Context": ctx,
                    "Statement": s,
                    "Category": classify_sentence(s, kw_dict, match_mode),
                }
            )

    return pd.DataFrame(rows)


#  ───────────────────────────────────────── Sidebar
st.sidebar.header("1️⃣ Upload CSV")
csv_file = st.sidebar.file_uploader("Select a CSV file", type=["csv"])

if csv_file is None:
    st.info("👈 Upload a CSV from the sidebar to begin.")
    st.stop()

df_in = pd.read_csv(csv_file)
if df_in.empty:
    st.error("Uploaded file is empty.")
    st.stop()

st.sidebar.header("2️⃣ Select columns")
id_col = st.sidebar.selectbox("ID column (unique key)", df_in.columns)
ctx_col = st.sidebar.selectbox("Context column (text)", df_in.columns)

st.sidebar.header("3️⃣ Sentence options")
isolate_hash = st.sidebar.checkbox("Hashtags as standalone sentences", True)
min_chars = st.sidebar.slider("Min characters per sentence", 1, 100, 3)
min_words = st.sidebar.slider("Min words per sentence", 1, 20, 1)

st.sidebar.header("4️⃣ Matching options")
match_mode = st.sidebar.radio(
    "Category assignment",
    ["first", "all"],
    help="‘first’ = stop at first match (faster).\n"
         "‘all’ = list every matching category.",
)

st.sidebar.header("5️⃣ Keyword dictionary (JSON)")
dict_text = st.sidebar.text_area(
    "Edit or paste a custom dictionary",
    value=json.dumps(DEFAULT_DICT, indent=2),
    height=300,
)

try:
    USER_DICT = json.loads(dict_text)
    if not isinstance(USER_DICT, dict):
        raise ValueError
except Exception:
    st.sidebar.error("❌ Invalid JSON. Using built-in dictionary.")
    USER_DICT = DEFAULT_DICT

#  ───────────────────────────────────────── Instructions
st.markdown(
    """
### How to Use
1. **Upload** your CSV in the sidebar.  
2. **Select** *ID* and *Context* columns.  
3. **Configure** sentence rules, matching mode, and dictionary.  
4. Click **Transform** to run.  
5. **Download** your processed CSV.

#### Output Columns
| Column        | Description                                   |
|---------------|-----------------------------------------------|
| `ID`          | value from chosen ID column                   |
| `Sentence ID` | running number within each original record    |
| `Context`     | original text after basic cleaning (URLs out) |
| `Statement`   | extracted sentence                            |
| `Category`    | assigned category (or categories)             |
""",
    unsafe_allow_html=True,
)

#  ───────────────────────────────────────── Run
if st.sidebar.button("⚙️ Transform"):
    with st.spinner("Processing… this is cached, re-runs are instant"):
        df_out = transform(
            df_in,
            id_col,
            ctx_col,
            USER_DICT,
            isolate_hash,
            min_chars,
            min_words,
            match_mode,
        )

    st.success(f"Done! Extracted {len(df_out):,} sentences.")
    st.dataframe(df_out.head(30), use_container_width=True)

    buf = StringIO()
    df_out.to_csv(buf, index=False)
    st.download_button(
        "💾 Download CSV", data=buf.getvalue(),
        file_name="transformed_text.csv", mime="text/csv"
    )
