# 📝 Text Transformation App – v2.1
# --- FIX: robust classify_sentence (no crash when no keywords match) ---

from __future__ import annotations
from io import StringIO
import json, re
from typing import List

import pandas as pd
import streamlit as st

# ───────────────────────── UI & CSS boilerplate (unchanged) ───────────────
st.set_page_config(
    page_title="Text Transformation App", page_icon="📝",
    layout="wide", initial_sidebar_state="expanded"
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

# ───────────────────────── Default keyword dictionary (same) ───────────────
DEFAULT_DICT = {
    "Fashion": ["style","fashion","wardrobe","clothes","outfit","ootd","runway",
                "trend","vogue","chic","couture","lookbook","garment","designer"],
    "Food":    ["delicious","foodie","dinner","lunch","restaurant","recipe","taste",
                "yummy","cuisine","dessert","snack","brunch","kitchen","chef","cook",
                "coffee","tea"],
    "Travel":  ["travel","trip","vacation","explore","journey","flight","hotel",
                "backpacking","adventure","wanderlust","tourism","passport",
                "roadtrip","getaway","destination"],
    "Fitness": ["workout","fitness","exercise","gym","training","cardio","strength",
                "run","yoga","hiit","wellness","fitspo","calisthenics","pilates",
                "cycling","marathon"],
    "Technology":["tech","gadget","smartphone","ai","machine learning","robot",
                "software","hardware","coding","programming","startup","innovation",
                "app","vr","blockchain","5g","cyber","cloud"],
    "Sports":  ["soccer","football","basketball","nba","nfl","cricket","baseball",
                "tennis","golf","olympics","athlete","score","goal","match",
                "tournament","league","championship","stadium"],
    "Beauty":  ["beauty","makeup","skincare","cosmetics","lipstick","foundation",
                "mascara","blush","palette","fragrance","haircare","salon","spa",
                "glow","nails","manicure"],
    "Nature":  ["nature","forest","mountain","beach","sunset","sunrise","wildlife",
                "ocean","lake","river","landscape","hiking","camping","outdoors",
                "flora","fauna","eco","green"],
    "Health":  ["health","nutrition","diet","vitamin","mental health","sleep",
                "mindfulness","meditation","doctor","medicine","healthy","hydration",
                "immune","well-being"],
    "Entertainment":["movie","film","cinema","music","concert","album","song","tv",
                "series","netflix","premiere","actor","actress","festival","show",
                "theatre","hollywood","bollywood"],
}

# ───────────────────────── Regex + helpers (unchanged except classify) ─────
URL_RE       = re.compile(r"https?://\S+")
PUNCT_ONLY_RE= re.compile(r"^[\W_]+$")
HASHTAG_RE   = re.compile(r"(#\w+)")

def clean_text(txt:str)->str:
    return URL_RE.sub("",txt).replace("\n"," ").strip()

def split_sentences(txt:str, isolate:bool)->List[str]:
    if not txt: return []
    if isolate: txt=HASHTAG_RE.sub(r". \1 .",txt)
    parts=re.split(r"(?<=[.!?])\s+|(?=#[^\s]+)",txt)
    return [p.strip() for p in parts if p.strip()]

def filter_sentence(s:str,min_chars:int,min_words:int)->bool:
    return (
        len(s)>=min_chars and
        len(s.split())>=min_words and
        not PUNCT_ONLY_RE.fullmatch(s)
    )

def classify_sentence(s:str, kw_dict:dict[str,List[str]], mode:str)->str:
    """
    SAFE version: never raises IndexError.
    mode='first'  -> first matching category or 'Uncategorized'
    mode='all'    -> ';'-joined list of every match or 'Uncategorized'
    """
    low=s.lower()
    hits=[cat for cat,kws in kw_dict.items() if any(k.lower() in low for k in kws)]
    if not hits:
        return "Uncategorized"
    return hits[0] if mode=="first" else ";".join(hits)

@st.cache_data(show_spinner=False)
def transform(df,id_col,ctx_col,kw,isolate,min_c,min_w,mode):
    rows=[]
    for _,r in df.iterrows():
        ctx=clean_text(str(r[ctx_col]))
        sents=split_sentences(ctx,isolate)
        idx=0
        for sent in sents:
            if not filter_sentence(sent,min_c,min_w): continue
            idx+=1
            rows.append({
                "ID":r[id_col],
                "Sentence ID":idx,
                "Context":ctx,
                "Statement":sent,
                "Category":classify_sentence(sent,kw,mode)
            })
    return pd.DataFrame(rows)

# ───────────────────────── Sidebar UI (unchanged) ──────────────────────────
st.sidebar.header("1️⃣ Upload CSV")
file=st.sidebar.file_uploader("Select a CSV file",type=["csv"])
if file is None:
    st.info("👈 Upload a CSV from the sidebar to begin."); st.stop()

df_in=pd.read_csv(file);  st.sidebar.header("2️⃣ Select columns")
id_col=st.sidebar.selectbox("ID column",df_in.columns)
ctx_col=st.sidebar.selectbox("Context column",df_in.columns)

st.sidebar.header("3️⃣ Sentence options")
isolate_hash=st.sidebar.checkbox("Hashtags standalone",True)
min_chars = st.sidebar.slider("Min chars",1,100,3)
min_words = st.sidebar.slider("Min words",1,20,1)

st.sidebar.header("4️⃣ Matching mode")
match_mode=st.sidebar.radio("Category assignment",["first","all"])

st.sidebar.header("5️⃣ Keyword dictionary (JSON)")
dict_text=st.sidebar.text_area("Edit JSON",json.dumps(DEFAULT_DICT,indent=2),height=300)
try:
    USER_DICT=json.loads(dict_text)
    if not isinstance(USER_DICT,dict): raise ValueError
except Exception:
    st.sidebar.error("❌ Invalid JSON – using defaults"); USER_DICT=DEFAULT_DICT

# ───────────────────────── Instructions (unchanged) ────────────────────────
st.markdown(
    """
### How to Use
1. **Upload** CSV → choose *ID*/*Context* columns  
2. **Options** → hashtags, filters, matching mode, dictionary  
3. Click **Transform** → preview & download

| Output column | Meaning |
|---------------|---------|
| ID | your chosen identifier |
| Sentence ID | running index within each record |
| Context | original cleaned text |
| Statement | extracted sentence |
| Category | assigned category |
""",
    unsafe_allow_html=True,
)

# ───────────────────────── Run transform ──────────────────────────
if st.sidebar.button("⚙️ Transform"):
    with st.spinner("Processing…"):
        df_out=transform(df_in,id_col,ctx_col,USER_DICT,
                         isolate_hash,min_chars,min_words,match_mode)
    st.success(f"Done! {len(df_out):,} sentences extracted.")
    st.dataframe(df_out.head(30),use_container_width=True)
    buf=StringIO(); df_out.to_csv(buf,index=False)
    st.download_button("💾 Download CSV",buf.getvalue(),
                       file_name="transformed_text.csv",mime="text/csv")
