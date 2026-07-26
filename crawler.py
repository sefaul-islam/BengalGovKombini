"""
Module A: Recursive Web Crawler
================================
Crawls government service websites (NID, ePassport, MyGov) starting from
seed URLs. Extracts main textual content, ignoring navbars/footers/scripts,
and saves structured JSON output.

Usage:
    python crawler.py                          # Crawl all default seed URLs
    python crawler.py --urls https://example.com --depth 3
    python crawler.py --output my_data.json
"""

import json
import re
import time
import logging
import argparse
from urllib.parse import urljoin, urlparse
from collections import deque
from typing import Optional

import requests
import urllib3
from bs4 import BeautifulSoup, Comment

# Suppress SSL warnings for government sites with misconfigured certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SEED_URLS: list[str] = [
    # Visa
    "https://bdvisainfo.com/visa-processing-documents/",
    # Passport
    "https://epassportbd.info/renew-step/",
    "https://www.epassport-bd.com/2022/04/passport-korte-ki-ki-lage.html",
    # NID
    "https://tourbd.info/nid-korte-ki-ki-lage/",
    "https://nidbdris.info/notun-voter-nibondhon-korte-ki-ki-lage/",
    # Birth Registration
    "https://eservicesbd.com/online-birth-registration-application/",
    # Marriage
    "https://reportbd.net/kazi-expense-for-marriage-bd/",
    "https://sparkadvocates.com/court-marriage-bangladesh-guide/",
    "https://marriagedivorcelawyerdhakabd.com/court-marriage-procedure-and-requirement-papers-in-bangladesh/",
    "https://marriagedivorcelawyerdhakabd.com/court-marriage-procedure-in-bangladesh/",
    "https://quickbangla.com/court-marriage-affidavit/",
]

DEFAULT_MAX_DEPTH = 2
REQUEST_TIMEOUT = 20  # seconds
DELAY_BETWEEN_REQUESTS = 1.0  # polite crawling delay
DEFAULT_OUTPUT_FILE = "scraped_data.json"

# Tags whose content is purely navigational / non-informational
STRIP_TAGS = [
    "nav", "footer", "header", "aside", "script", "style", "noscript",
    "iframe", "svg", "form", "button", "input", "select", "textarea",
]

# CSS classes / IDs commonly used for navigation elements
NAV_PATTERNS = re.compile(
    r"(navbar|nav-bar|site-header|site-footer|footer|sidebar|menu|breadcrumb|"
    r"cookie|popup|modal|advertisement|ad-banner|social-share|share-btn)",
    re.IGNORECASE,
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "bn-BD,bn;q=0.9,en;q=0.8",
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def is_same_domain(base_url: str, target_url: str) -> bool:
    """Check whether target_url belongs to the same domain as base_url."""
    base_domain = urlparse(base_url).netloc
    target_domain = urlparse(target_url).netloc
    return target_domain == base_domain


def clean_text(text: str) -> str:
    """Normalise whitespace and strip stray characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_content(html: str, url: str) -> dict[str, str]:
    """
    Parse *html* and return {"url", "title", "content"} with main textual
    body.  Navigation / footer / script nodes are stripped.
    """
    soup = BeautifulSoup(html, "lxml")

    # --- Remove non-content tags ---
    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # --- Remove HTML comments ---
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # --- Remove elements whose class/id looks navigational ---
    for element in soup.find_all(True):
        if element.attrs is None:
            continue
        classes = " ".join(element.get("class", []))
        el_id = element.get("id", "") or ""
        if NAV_PATTERNS.search(classes) or NAV_PATTERNS.search(el_id):
            element.decompose()

    # --- Extract title ---
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # --- Try to isolate <main> or <article>, fall back to <body> ---
    main = soup.find("main") or soup.find("article") or soup.find("body")
    if main is None:
        main = soup  # last resort

    raw_text = main.get_text(separator="\n", strip=True)
    content = clean_text(raw_text)

    return {"url": url, "title": title, "content": content}


def fetch_page(url: str, session: requests.Session) -> Optional[str]:
    """GET *url* and return the HTML string, or None on failure."""
    try:
        resp = session.get(
            url, headers=HEADERS, timeout=REQUEST_TIMEOUT, verify=False
        )
        resp.raise_for_status()
        # Only process HTML responses
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            logger.debug("Skipping non-HTML: %s (%s)", url, content_type)
            return None
        resp.encoding = resp.apparent_encoding  # handle Bengali encoding
        return resp.text
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def discover_links(html: str, base_url: str) -> list[str]:
    """Return de-duplicated absolute URLs found on the page (query params stripped)."""
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Skip anchors, javascript, mailto, tel
        if href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absolute = urljoin(base_url, href)
        # Remove fragment and query parameters to avoid duplicate pages
        parsed = urlparse(absolute)
        absolute = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        # Skip non-HTML file extensions
        skip_exts = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".mp4", ".zip", ".doc", ".docx", ".xls")
        if any(absolute.lower().endswith(ext) for ext in skip_exts):
            continue
        if absolute not in links:
            links.append(absolute)
    return links


# ---------------------------------------------------------------------------
# Main Crawler
# ---------------------------------------------------------------------------
def crawl(
    seed_urls: list[str],
    max_depth: int = DEFAULT_MAX_DEPTH,
    output_file: str = DEFAULT_OUTPUT_FILE,
) -> list[dict[str, str]]:
    """
    BFS crawl starting from *seed_urls* up to *max_depth* levels deep.
    Returns list of {"url", "title", "content"} dicts and writes them to
    *output_file*.
    """
    visited: set[str] = set()
    results: list[dict[str, str]] = []
    queue: deque[tuple[str, str, int]] = deque()
    # queue items: (url, seed_domain_url, depth)

    for seed in seed_urls:
        queue.append((seed, seed, 0))

    session = requests.Session()

    logger.info(
        "Starting crawl with %d seed URL(s), max depth=%d", len(seed_urls), max_depth
    )

    while queue:
        url, seed_url, depth = queue.popleft()

        if url in visited:
            continue
        if depth > max_depth:
            continue

        visited.add(url)
        logger.info("[depth=%d] Crawling: %s", depth, url)

        html = fetch_page(url, session)
        if html is None:
            continue

        page_data = extract_content(html, url)

        # Only keep pages with meaningful content (> 50 chars)
        if len(page_data["content"]) > 50:
            results.append(page_data)
            logger.info(
                "  ✓ Extracted %d chars from: %s",
                len(page_data["content"]),
                page_data["title"] or url,
            )
        else:
            logger.debug("  ✗ Skipped (too little content): %s", url)

        # Discover child links (only within same domain)
        if depth < max_depth:
            child_links = discover_links(html, url)
            for link in child_links:
                if link not in visited and is_same_domain(seed_url, link):
                    queue.append((link, seed_url, depth + 1))

        time.sleep(DELAY_BETWEEN_REQUESTS)

    # --- Write output ---
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    logger.info(
        "Crawl complete. %d pages saved to %s", len(results), output_file
    )
    return results


# ---------------------------------------------------------------------------
# CLI Entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursive web crawler for Bangladeshi government service sites."
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        default=DEFAULT_SEED_URLS,
        help="Seed URLs to start crawling from.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help=f"Maximum crawl depth (default: {DEFAULT_MAX_DEPTH}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT_FILE}).",
    )
    args = parser.parse_args()

    crawl(seed_urls=args.urls, max_depth=args.depth, output_file=args.output)


if __name__ == "__main__":
    main()
