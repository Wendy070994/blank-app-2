# ─── Emoji-safe helpers ─────────────────────────────────────────────
import re
import emoji

def _remove_emoji(text: str) -> str:
    """
    Strip all emoji from *text*.

    • Uses emoji.replace_emoji() when the function exists (emoji ≥ 2.0.0).
    • Falls back to a regex that matches any code-point with
      the Emoji presentation property (works for older versions).
    """
    if not text or not isinstance(text, str):
        return text

    if hasattr(emoji, "replace_emoji"):          # emoji ≥ 2.0.0
        return emoji.replace_emoji(text, replace="")

    # — Fallback for emoji < 2.0.0 —
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001F64F"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"   # transport & map
        "\U0001F700-\U0001F77F"   # alchemical symbols
        "\U0001F780-\U0001F7FF"   # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"   # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"   # Supplemental Symbols & Pictographs
        "\U0001FA00-\U0001FAFF"   # Symbols & Pictographs Extended-A
        "\U00002702-\U000027B0"   # Dingbats
        "\U000024C2-\U0001F251"   # Enclosed characters
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)
