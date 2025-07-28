# app.py
"""
Instagram Caption Transformer – Streamlit Edition
-------------------------------------------------
Upload a CSV of Instagram posts, clean the captions,
split them into sentences, and download the transformed file.
"""

import csv
import logging
import re
from typing import List

import pandas as pd
import streamlit as st
from unidecode import unidecode

# One-time NLTK download (safe to re-run)
nltk.download("punkt", quiet=True)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helper class
# ---------------------------------------------------------------------
class TextPreprocessor:
    """Handles all cleaning / splitting for Instagram captions."""

    def __init__(self):
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]"
            r"|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self.email_pattern = re.compile(r"\S+@\S+")
        self.hashtag_pattern = re.compile(r"#[\w\d]+")
        self.mention_pattern = re.compile(r"@[\w\d]+")

    # ---- Cleaning helpers -------------------------------------------------
    @staticmethod
    def remove_emoji(text: str) -> str:
        return emoji.replace_emoji(text, replace="")

    def clean_text(self, text: str) -> str:
        """Full caption cleaning pipeline."""
        if not isinstance(text, str):
            return "[PAD]"

        # Remove emojis first
        text = self.remove_emoji(text)

        # Normalize accented chars
        text = unidecode(text)

        # Strip URLs / emails / tags / mentions
        text = self.url_pattern.sub("", text)
        text = self.email_pattern.sub("", text)
        text = self.hashtag_pattern.sub("", text)
        text = self.mention_pattern.sub("", text)

        # Collapse whitespace / newlines
        text = " ".join(text.replace("\n", " ").split())

        return text if text.strip() else "[PAD]"

    # ---- Sentence splitting ----------------------------------------------
    def split_sentences(self, caption: str) -> List[str]:
        """Return cleaned sentences from one caption."""
        if caption and caption[-1] not in ".!?":
            caption += "."
        sentences = sent_tokenize(caption)
        return [s.strip() for s in sentences if s.strip()]

    # ---- Data-frame transformation ---------------------------------------
    def transform_caption_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate one row per sentence w/ required columns."""
        transformed = []

        for _, row in df.iterrows():
            if pd.isna(row["caption"]):
                continue

            sentences = self.split_sentences(row["cleaned_caption"])
            for turn, sentence in enumerate(sentences, 1):
                transformed.append(
                    {
                        "shortcode": row.get("shortcode", row.get("post_id", "")),
                        "turn": turn,
                        "caption": row["caption"],
                        "transcript": sentence,
                        "post_url": row.get("post_url", ""),
                    }
                )

        return pd.DataFrame(transformed)


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
def main() -> None:
    st.title("📄➡️ Instagram Caption Transformer")
    st.write(
        "Upload a **CSV** containing at least a `caption` column. "
        "The app will clean each caption, split it into sentences, "
        "and return one row per sentence."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"], accept_multiple_files=False
    )

    if uploaded_file is None:
        st.info("👈 Drag a CSV here or click to browse.")
        st.stop()

    # Read CSV (Streamlit returns a file-like buffer)
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    if "caption" not in df.columns:
        st.error("The file must contain a column named **caption**.")
        st.stop()

    # Clean & transform
    pre = TextPreprocessor()
    df["cleaned_caption"] = df["caption"].apply(pre.clean_text)
    output_df = pre.transform_caption_df(df)

    # Show preview
    st.success(f"Created {len(output_df)} sentence-level rows.")
    st.dataframe(output_df.head(), use_container_width=True)

    # Download button
    csv_bytes = output_df.to_csv(
        index=False, quoting=csv.QUOTE_NONNUMERIC
    ).encode("utf-8")
    st.download_button(
        "Download transformed CSV",
        data=csv_bytes,
        file_name="ig_posts_transformed.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

