#!/usr/bin/env python3
from __future__ import annotations

"""
Wikipedia Medical Entries Scraper

Goal:
  - Discover medical-related article pages from Wikipedia category graph
  - Fetch plaintext extracts for each article
  - Save incrementally with resume support

Default output:
  ../../Dataset/02_Unstructured/wikipedia_medical_100k.jsonl
"""

import argparse
import json
import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_SEED_CATEGORIES = [
    "Medicine",
    "Diseases and disorders",
    "Health",
    "Medical treatments",
    "Human anatomy",
    "Pharmacology",
]


class WikipediaMedicalScraper:
    def __init__(
        self,
        language: str,
        target: int,
        output_file: Path,
        progress_file: Path,
        delay: float,
        max_retries: int,
        batch_size: int,
        checkpoint_every: int,
        seed_categories: list[str],
    ):
        self.language = language
        self.target = target
        self.output_file = output_file
        self.progress_file = progress_file
        self.delay = delay
        self.max_retries = max_retries
        self.batch_size = max(1, min(batch_size, 50))
        self.checkpoint_every = max(1, checkpoint_every)
        self.seed_categories = seed_categories

        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.base_page_url = f"https://{language}.wikipedia.org/wiki/"
        self.headers = {
            "User-Agent": "MedicalDataScraper/1.0 (academic use; contact: local)",
            "Accept": "application/json",
        }

        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if self.progress_file.exists():
            with open(self.progress_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            logger.info(
                "Loaded progress: %d discovered pages, %d written pages, %d pending categories",
                len(state.get("discovered_pages", {})),
                len(state.get("written_page_ids", [])),
                len(state.get("category_queue", [])),
            )
            return state

        state = {
            "language": self.language,
            "target": self.target,
            "seed_categories": self.seed_categories,
            "category_queue": self.seed_categories.copy(),
            "visited_categories": [],
            "discovered_pages": {},  # pageid(str) -> {"title": ..., "source_category": ...}
            "written_page_ids": [],
            "last_updated": time.time(),
        }
        return state

    def _save_state(self):
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.state["last_updated"] = time.time()
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def _api_get(self, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(
                    self.api_url,
                    params=params,
                    headers=self.headers,
                    timeout=40,
                )
                resp.raise_for_status()
                payload = resp.json()
                if "error" in payload:
                    raise RuntimeError(f"Wikipedia API error: {payload['error']}")
                return payload
            except Exception as exc:
                if attempt == self.max_retries:
                    raise
                wait = min(2**attempt, 30)
                logger.warning("Request failed (%s), retry in %ss", exc, wait)
                time.sleep(wait)

    def _normalize_seed_categories(self) -> list[str]:
        normalized = []
        for cat in self.seed_categories:
            c = cat.strip()
            if not c:
                continue
            if c.startswith("Category:"):
                c = c[len("Category:") :]
            normalized.append(c)
        return normalized

    def discover_pages(self):
        self.state["category_queue"] = [
            c[len("Category:") :] if c.startswith("Category:") else c
            for c in self.state.get("category_queue", [])
        ]
        if not self.state["category_queue"]:
            self.state["category_queue"] = self._normalize_seed_categories()

        queue = deque(self.state["category_queue"])
        visited = set(self.state.get("visited_categories", []))
        discovered = self.state.get("discovered_pages", {})

        logger.info("Start category traversal to discover up to %d pages", self.target)
        logger.info(
            "Current status: %d discovered, %d categories queued, %d categories visited",
            len(discovered),
            len(queue),
            len(visited),
        )

        steps = 0
        while queue and len(discovered) < self.target:
            category_name = queue.popleft().strip()
            if not category_name or category_name in visited:
                continue

            visited.add(category_name)
            steps += 1
            logger.info(
                "[CAT %d] Category:%s | discovered=%d | queue=%d",
                steps,
                category_name,
                len(discovered),
                len(queue),
            )

            cmcontinue = None
            while len(discovered) < self.target:
                params = {
                    "action": "query",
                    "format": "json",
                    "list": "categorymembers",
                    "cmtitle": f"Category:{category_name}",
                    "cmtype": "page|subcat",
                    "cmlimit": "500",
                }
                if cmcontinue:
                    params["cmcontinue"] = cmcontinue

                payload = self._api_get(params)
                members = payload.get("query", {}).get("categorymembers", [])
                if not members:
                    break

                for m in members:
                    ns = m.get("ns")
                    title = m.get("title", "")
                    pageid = m.get("pageid")

                    # Namespace 0: article page
                    if ns == 0 and pageid is not None:
                        key = str(pageid)
                        if key not in discovered:
                            discovered[key] = {
                                "title": title,
                                "source_category": category_name,
                            }
                            if len(discovered) % 1000 == 0:
                                logger.info("Discovered %d pages", len(discovered))
                            if len(discovered) >= self.target:
                                break

                    # Namespace 14: category page
                    elif ns == 14 and title.startswith("Category:"):
                        subcat = title[len("Category:") :]
                        if subcat not in visited:
                            queue.append(subcat)

                if len(discovered) >= self.target:
                    break

                cmcontinue = payload.get("continue", {}).get("cmcontinue")
                if not cmcontinue:
                    break

                time.sleep(self.delay)

            self.state["discovered_pages"] = discovered
            self.state["visited_categories"] = sorted(visited)
            self.state["category_queue"] = list(queue)
            self._save_state()
            time.sleep(self.delay)

        logger.info(
            "Discovery complete: %d pages discovered, %d categories still queued",
            len(discovered),
            len(queue),
        )

    def _fetch_extracts_batch(self, page_ids: list[str]) -> dict[str, dict[str, Any]]:
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "pageids": "|".join(page_ids),
            "explaintext": 1,
            "exintro": 1,
            "exsectionformat": "plain",
            "exlimit": "max",
        }
        out: dict[str, dict[str, Any]] = {}

        while True:
            payload = self._api_get(params)
            pages = payload.get("query", {}).get("pages", {})
            for pid, page_obj in pages.items():
                key = str(pid)
                prev = out.get(key, {})
                prev_extract = (prev.get("extract") or "").strip()
                new_extract = (page_obj.get("extract") or "").strip()

                # Keep richer object to avoid continuation pages wiping extract.
                if prev and prev_extract and not new_extract:
                    merged = dict(page_obj)
                    merged["extract"] = prev.get("extract", "")
                    if not merged.get("title") and prev.get("title"):
                        merged["title"] = prev["title"]
                    out[key] = merged
                else:
                    out[key] = page_obj

            cont = payload.get("continue")
            if not cont:
                break
            params.update(cont)

        return out

    @staticmethod
    def _clean_extract(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\xa0", " ")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fetch_and_write(self):
        discovered = self.state.get("discovered_pages", {})
        if not discovered:
            logger.warning("No discovered pages found, skip content fetching")
            return

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        written = set(self.state.get("written_page_ids", []))
        pending_ids = [pid for pid in discovered.keys() if pid not in written][: self.target]
        total_to_write = len(pending_ids)

        logger.info(
            "Start content fetch: %d already written, %d pending",
            len(written),
            total_to_write,
        )

        if total_to_write == 0:
            logger.info("All discovered pages already written")
            return

        with open(self.output_file, "a", encoding="utf-8") as f:
            batch_count = 0
            for i in range(0, total_to_write, self.batch_size):
                batch_ids = pending_ids[i : i + self.batch_size]
                data = self._fetch_extracts_batch(batch_ids)

                for pid in batch_ids:
                    meta = discovered.get(pid, {})
                    page_obj = data.get(pid, {})
                    title = page_obj.get("title") or meta.get("title", "")
                    extract = self._clean_extract(page_obj.get("extract", "") or "")
                    fullurl = page_obj.get("fullurl")
                    if not fullurl and title:
                        fullurl = self.base_page_url + quote(title.replace(" ", "_"))

                    record = {
                        "page_id": int(pid),
                        "title": title,
                        "url": fullurl or "",
                        "source_category": meta.get("source_category", ""),
                        "extract": extract,
                        "language": self.language,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written.add(pid)

                batch_count += 1
                if batch_count % self.checkpoint_every == 0:
                    self.state["written_page_ids"] = list(written)
                    self._save_state()

                logger.info(
                    "Wrote %d/%d pages",
                    min(i + len(batch_ids), total_to_write),
                    total_to_write,
                )
                time.sleep(self.delay)

        self.state["written_page_ids"] = list(written)
        self._save_state()
        logger.info("Done. Output saved to %s", self.output_file)

    def run(self):
        if len(self.state.get("discovered_pages", {})) < self.target:
            self.discover_pages()
        else:
            logger.info("Skip discovery: enough pages already discovered")

        self.fetch_and_write()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape up to N medical-related Wikipedia entries with resume support."
    )
    parser.add_argument(
        "--target",
        type=int,
        default=100_000,
        help="Target number of pages to collect (default: 100000).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Wikipedia language code (default: en).",
    )
    parser.add_argument(
        "--seed-categories",
        type=str,
        default=",".join(DEFAULT_SEED_CATEGORIES),
        help="Comma-separated seed categories (without Category: prefix).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(Path(__file__).resolve().parents[2] / "Dataset" / "02_Unstructured" / "wikipedia_medical_100k.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=(Path(__file__).resolve().parents[2] / "Dataset" / "02_Unstructured" / ".wikipedia_medical_progress.json"),
        help="Progress file path.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between API calls in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for each API request.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for page extract requests (max 50 for API safety).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save progress every N extract batches.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_categories = [s.strip() for s in args.seed_categories.split(",") if s.strip()]

    scraper = WikipediaMedicalScraper(
        language=args.lang,
        target=args.target,
        output_file=args.output,
        progress_file=args.progress,
        delay=args.delay,
        max_retries=args.max_retries,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        seed_categories=seed_categories,
    )
    scraper.run()


if __name__ == "__main__":
    main()
