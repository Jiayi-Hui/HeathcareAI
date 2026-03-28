#!/usr/bin/env python3
"""
NHS Health A-Z Scraper
Scrapes entries from four NHS index pages:
  - Conditions A-Z    https://www.nhs.uk/conditions/
  - Symptoms A-Z      https://www.nhs.uk/symptoms/
  - Tests & treatments https://www.nhs.uk/tests-and-treatments/
  - Medicines A-Z     https://www.nhs.uk/medicines/

Output: one JSON file per category in ../data/
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import logging
from pathlib import Path
from urllib.parse import urljoin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.nhs.uk"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "Dataset" / "02_Unstructured"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}
DELAY = 1.5        # seconds between detail-page requests
MAX_RETRIES = 3

# The four categories to scrape
CATEGORIES = {
    "conditions": {
        "index_url": f"{BASE_URL}/conditions/",
        "link_pattern": re.compile(r"^/conditions/[a-z]"),
        "output_file": "nhs_conditions.json",
    },
    "symptoms": {
        "index_url": f"{BASE_URL}/symptoms/",
        "link_pattern": re.compile(r"^/symptoms/[a-z]"),
        "output_file": "nhs_symptoms.json",
    },
    "tests-and-treatments": {
        "index_url": f"{BASE_URL}/tests-and-treatments/",
        "link_pattern": re.compile(r"^/tests-and-treatments/[a-z]"),
        "output_file": "nhs_tests_and_treatments.json",
    },
    "medicines": {
        "index_url": f"{BASE_URL}/medicines/",
        "link_pattern": re.compile(r"^/medicines/[a-z]"),
        "output_file": "nhs_medicines.json",
    },
}


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
# Step 1 – collect entry links from an NHS index page
# ---------------------------------------------------------------------------

def get_entry_links(index_url: str, link_pattern: re.Pattern) -> list[dict]:
    """
    Return a list of {name, url} dicts for every entry link on the index page.
    NHS index pages may span multiple pages (pagination) – we handle that too.
    """
    all_links: dict[str, str] = {}   # url -> name  (dedup)
    page_url = index_url

    while page_url:
        logger.info(f"  Fetching index page: {page_url}")
        resp = fetch(page_url)
        soup = BeautifulSoup(resp.text, "lxml")

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip().rstrip("/")
            if not link_pattern.match(href):
                continue
            # Skip links that are just the index itself
            if href.rstrip("/") in (
                "/conditions", "/symptoms",
                "/tests-and-treatments", "/medicines",
            ):
                continue
            full_url = urljoin(BASE_URL, href)
            if full_url not in all_links:
                name = a_tag.get_text(strip=True) or href.split("/")[-1]
                all_links[full_url] = name

        # Check for a "Next" pagination link
        next_link = soup.select_one('a[rel="next"]') or soup.select_one('li.nhsuk-pagination__item--next a')
        if next_link and next_link.get("href"):
            page_url = urljoin(BASE_URL, next_link["href"])
            time.sleep(0.5)
        else:
            page_url = None

    entries = [{"name": name, "url": url} for url, name in all_links.items()]
    entries.sort(key=lambda e: e["name"].lower())
    return entries


# ---------------------------------------------------------------------------
# Step 2 – extract content from a single NHS entry page
# ---------------------------------------------------------------------------

def extract_entry_content(soup: BeautifulSoup) -> str:
    """
    Extract the main textual content from an NHS entry page.
    Returns cleaned plain text.
    """
    # NHS pages put main content in <main id="maincontent"> or <article>
    main = soup.select_one("main#maincontent") or soup.select_one("main") or soup.select_one("article")
    if not main:
        main = soup

    # Remove navigation, breadcrumbs, pagination, sidebars, related links
    for tag in main.select(
        "nav, .nhsuk-breadcrumb, .nhsuk-pagination, "
        ".nhsuk-related-nav, .nhsuk-contents-list, "
        "script, style, header, footer, .nhsuk-action-link"
    ):
        tag.decompose()

    sections: list[str] = []

    # Iterate over headings and paragraphs to build structured text
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        tag_name = el.name
        text = el.get_text(strip=True)
        if not text:
            continue

        if tag_name in ("h1", "h2"):
            sections.append(f"\n## {text}\n")
        elif tag_name in ("h3", "h4"):
            sections.append(f"\n### {text}\n")
        elif tag_name == "li":
            sections.append(f"- {text}")
        elif tag_name == "table":
            sections.append(_table_to_text(el))
        else:
            sections.append(text)

    return "\n".join(sections).strip()


def _table_to_text(table_tag) -> str:
    """Simple table → text conversion."""
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _progress_path(category: str) -> Path:
    return OUTPUT_DIR / f".nhs_{category}_progress.json"


def load_progress(category: str) -> dict:
    p = _progress_path(category)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"done": {}}


def save_progress(category: str, progress: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_progress_path(category), "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Scrape one category
# ---------------------------------------------------------------------------

def scrape_category(cat_key: str, cat_cfg: dict):
    """Scrape all entries for one NHS category."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Category: {cat_key}")
    logger.info(f"{'='*60}")

    entries = get_entry_links(cat_cfg["index_url"], cat_cfg["link_pattern"])
    logger.info(f"Found {len(entries)} entries for [{cat_key}]")

    progress = load_progress(cat_key)
    done: dict = progress.get("done", {})

    total = len(entries)
    for idx, entry in enumerate(entries, 1):
        url = entry["url"]
        if url in done:
            logger.info(f"  [{idx}/{total}] Skip (cached): {entry['name']}")
            continue

        logger.info(f"  [{idx}/{total}] Scraping: {entry['name']}  →  {url}")
        try:
            resp = fetch(url)
            soup = BeautifulSoup(resp.text, "lxml")

            # Grab page title from <h1> if available
            h1 = soup.select_one("h1")
            title = h1.get_text(strip=True) if h1 else entry["name"]

            content = extract_entry_content(soup)

            done[url] = {
                "category": cat_key,
                "title": title,
                "name": entry["name"],
                "url": url,
                "content": content,
            }
            progress["done"] = done
            save_progress(cat_key, progress)

            if not content:
                logger.warning(f"    ⚠ No content extracted")

        except Exception as exc:
            logger.error(f"    ✗ Error: {exc}")
            done[url] = {
                "category": cat_key,
                "title": entry["name"],
                "name": entry["name"],
                "url": url,
                "content": "",
                "error": str(exc),
            }
            progress["done"] = done
            save_progress(cat_key, progress)

        time.sleep(DELAY)

    # --- Write final output -----------------------------------------------
    results = sorted(done.values(), key=lambda r: r["name"].lower())
    out_file = OUTPUT_DIR / cat_cfg["output_file"]
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results if r.get("content"))
    logger.info(f"[{cat_key}] Done!  {ok}/{len(results)} entries with content  →  {out_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cat_key, cat_cfg in CATEGORIES.items():
        scrape_category(cat_key, cat_cfg)

    logger.info("\n✅  All NHS categories scraped!")


if __name__ == "__main__":
    main()
