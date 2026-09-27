"""
fetch_wikipedia.py
------------------
Downloads Telugu Wikipedia articles using the MediaWiki API,
cleans the HTML with BeautifulSoup, and saves each article
as a UTF-8 text file in the data/ directory.

Run:
    python fetch_wikipedia.py
"""

import os
import re
import time

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base URL for the Telugu Wikipedia API
WIKIPEDIA_API = "https://te.wikipedia.org/w/api.php"

# Output directory for article text files
DATA_DIR = "data"

# Target Telugu Wikipedia article titles to fetch
ARTICLES = [
    "తమిళనాడు",
    "తెలంగాణ",
    "ఆంధ్రప్రదేశ్",
    "రుద్రమ దేవి",
    "కాకతీయ సామ్రాజ్యం",
    "శ్రీ కృష్ణదేవరాయలు",
    "నన్నయ",
    "తిక్కన",
    "పోతన",
    "అమరావతి",
    "వరంగల్",
    "విజయనగర సామ్రాజ్యం",
]

# Polite delay between requests (seconds) to avoid rate-limiting
REQUEST_DELAY = 1.5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_article_html(title: str) -> str | None:
    """
    Calls the MediaWiki 'parse' action to retrieve rendered HTML
    for a given article title from Telugu Wikipedia.

    Args:
        title: The Telugu Wikipedia article title.

    Returns:
        Raw HTML string, or None if the request fails.
    """
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "redirects": True,   # follow redirects automatically
    }
    headers = {
        # Identify the bot politely per Wikimedia guidelines
        "User-Agent": "TeluguRAG/2.0 (educational project; contact: user@example.com)"
    }

    try:
        response = requests.get(WIKIPEDIA_API, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()

        # Check for API-level errors (e.g., page not found)
        if "error" in data:
            print(f"  [API ERROR] {title}: {data['error'].get('info', 'unknown error')}")
            return None

        return data["parse"]["text"]["*"]

    except requests.RequestException as exc:
        print(f"  [NETWORK ERROR] {title}: {exc}")
        return None


def clean_html(html: str) -> str:
    """
    Strips HTML tags, navigation noise, references, and edit-section links
    from Wikipedia article HTML, returning plain Telugu text.

    Args:
        html: Raw Wikipedia HTML string.

    Returns:
        Clean plain-text string.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements: tables of contents, navboxes,
    # reference sections, edit links, and infoboxes
    for tag in soup.find_all(
        class_=[
            "toc",           # table of contents
            "navbox",        # navigation box at page bottom
            "reflist",       # reference list
            "mw-editsection",# [edit] links
            "infobox",       # structured infobox on the right
            "sistersitebox", # links to sister projects
            "hatnote",       # disambiguation notes
            "thumb",         # image thumbnails + captions
        ]
    ):
        tag.decompose()

    # Also remove <sup> (footnote numbers) and <style> blocks
    for tag in soup.find_all(["sup", "style", "script"]):
        tag.decompose()

    # Extract text; use newline as paragraph separator
    text = soup.get_text(separator="\n")

    # Collapse runs of 3+ blank lines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading / trailing whitespace on each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines).strip()

    return text


def title_to_filename(title: str) -> str:
    """
    Converts a Telugu article title to a safe ASCII-safe filename.
    Non-ASCII characters are kept; only filesystem-unsafe characters
    (slashes, colons, etc.) are replaced with underscores.

    Args:
        title: Article title string.

    Returns:
        Filename string ending with .txt
    """
    # Replace characters that are unsafe in filenames
    safe = re.sub(r'[\\/:*?"<>|]', "_", title)
    # Collapse multiple spaces/underscores
    safe = re.sub(r"[\s_]+", "_", safe).strip("_")
    return f"{safe}.txt"


def save_article(title: str, text: str) -> str:
    """
    Saves cleaned article text to the data/ directory as a UTF-8 file.

    Args:
        title: Article title (used to derive filename).
        text: Cleaned plain text to write.

    Returns:
        Path of the saved file.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = title_to_filename(title)
    filepath = os.path.join(DATA_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        # Write a header line so the ingestion pipeline can extract the title
        f.write(f"శీర్షిక: {title}\n\n")   # "Title:" in Telugu
        f.write(text)

    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Iterates over ARTICLES, fetches each from Telugu Wikipedia,
    cleans the HTML, and saves it to the data/ directory.
    """
    print("=" * 60)
    print("Telugu Wikipedia Article Fetcher")
    print("=" * 60)
    print(f"Target: {len(ARTICLES)} articles → {DATA_DIR}/\n")

    success_count = 0
    fail_count = 0

    for index, title in enumerate(ARTICLES, start=1):
        print(f"[{index:02d}/{len(ARTICLES)}] Fetching: {title}")

        # Fetch raw HTML from Wikipedia API
        html = fetch_article_html(title)
        if html is None:
            fail_count += 1
            continue

        # Clean HTML to plain text
        text = clean_html(html)

        if len(text) < 100:
            # Suspiciously short — likely a redirect or empty page
            print(f"  [SKIP] Article too short ({len(text)} chars), skipping.")
            fail_count += 1
            continue

        # Save to data/ directory
        filepath = save_article(title, text)
        char_count = len(text)
        print(f"  [OK] Saved → {filepath}  ({char_count:,} chars)")
        success_count += 1

        # Polite delay between requests
        if index < len(ARTICLES):
            time.sleep(REQUEST_DELAY)

    print()
    print("=" * 60)
    print(f"Done. Success: {success_count}  |  Failed: {fail_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
