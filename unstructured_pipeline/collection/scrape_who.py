#!/usr/bin/env python3
"""
WHO Health Topics Scraper
Scrapes overview text for all health topics from https://www.who.int/health-topics
Output: ../data/who_health_topics.json
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.who.int"
INDEX_URL = f"{BASE_URL}/health-topics"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "Dataset" / "02_Unstructured"
OUTPUT_FILE = OUTPUT_DIR / "who_health_topics.json"
PROGRESS_FILE = OUTPUT_DIR / ".who_progress.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
DELAY = 2          # seconds between detail-page requests
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Networking helpers
# ---------------------------------------------------------------------------

def fetch(url: str, retries: int = MAX_RETRIES) -> requests.Response:
    """GET with retries and exponential back-off."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            logger.warning(f"  Attempt {attempt} failed ({exc}), retrying in {wait}s …")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Step 1 – collect all topic URLs from the index page
# ---------------------------------------------------------------------------

def get_topic_links() -> dict:
    """Return {slug: {name, url}} for every health topic on the index page."""
    logger.info(f"Fetching topic index: {INDEX_URL}")
    resp = fetch(INDEX_URL)
    soup = BeautifulSoup(resp.text, "lxml")

    topics: dict = {}
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        # Match both relative (/health-topics/slug) and absolute URLs
        m = re.match(
            r"^(?:https://www\.who\.int)?/health-topics/([a-z][a-z0-9\-]*)/?$",
            href,
        )
        if not m:
            continue
        slug = m.group(1)
        if slug in topics:
            continue
        # Prefer aria-label (clean name) over inner text (includes category)
        name = a_tag.get("aria-label") or ""
        if not name:
            p_heading = a_tag.select_one("p.heading")
            name = p_heading.get_text(strip=True) if p_heading else ""
        if not name:
            name = a_tag.get_text(strip=True) or slug
        topics[slug] = {"name": name, "url": f"{BASE_URL}/health-topics/{slug}"}

    logger.info(f"Found {len(topics)} unique topics")
    return topics


# ---------------------------------------------------------------------------
# Step 2 – extract overview from a single topic page
# ---------------------------------------------------------------------------

def extract_overview(soup: BeautifulSoup) -> str:
    """
    Try several strategies to pull the 'Overview' prose from a WHO topic page.
    Returns plain-text paragraphs joined by double newlines.
    """

    # --- Strategy A: look for the Overview tab panel ----------------------
    # WHO pages wrap the first tab's content in <div id="PageContent_...">
    # or a div whose data-defined attribute mentions 'Overview'.
    for sel in (
        'div[id*="Overview"]',
        'div[id*="overview"]',
        'section[id*="Overview"]',
        'div.section-overview',
    ):
        node = soup.select_one(sel)
        if node:
            text = _paras(node)
            if text:
                return text

    # --- Strategy B: first .sf-content-block that holds real paragraphs ---
    for block in soup.select(".sf-content-block"):
        text = _paras(block)
        if text and len(text) > 100:
            return text

    # --- Strategy C: <article> or <main> ----------------------------------
    for sel in ("article", "main", "#content"):
        node = soup.select_one(sel)
        if node:
            text = _paras(node)
            if text:
                return text

    # --- Strategy D: just grab all long paragraphs on the page ------------
    text = _paras(soup, min_len=60)
    return text


def _paras(root, min_len: int = 20) -> str:
    """Join <p> text from *root*, filtering short noise paragraphs."""
    paragraphs = root.find_all("p")
    lines = []
    for p in paragraphs:
        t = p.get_text(strip=True)
        if len(t) >= min_len:
            lines.append(t)
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Progress helpers (resume-safe)
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"done": {}}


def save_progress(progress: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    topics = get_topic_links()
    progress = load_progress()
    done: dict = progress.get("done", {})

    total = len(topics)
    for idx, (slug, info) in enumerate(topics.items(), 1):
        url = info["url"]
        if slug in done:
            logger.info(f"[{idx}/{total}] Skip (cached): {info['name']}")
            continue

        logger.info(f"[{idx}/{total}] Scraping: {info['name']}  →  {url}")
        try:
            resp = fetch(url)
            soup = BeautifulSoup(resp.text, "lxml")
            overview = extract_overview(soup)

            done[slug] = {
                "topic": info["name"],
                "slug": slug,
                "url": url,
                "overview": overview,
            }
            progress["done"] = done
            save_progress(progress)

            if not overview:
                logger.warning(f"  ⚠ No overview text extracted for {slug}")

        except Exception as exc:
            logger.error(f"  ✗ Error scraping {url}: {exc}")
            done[slug] = {
                "topic": info["name"],
                "slug": slug,
                "url": url,
                "overview": "",
                "error": str(exc),
            }
            progress["done"] = done
            save_progress(progress)

        time.sleep(DELAY)

    # --- Write final output -----------------------------------------------
    results = list(done.values())
    results.sort(key=lambda r: r["slug"])
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results if r.get("overview"))
    logger.info(f"Done!  {ok}/{len(results)} topics with overview text  →  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
