"""
Generic NewHomeSource review scraper.

Auto-detects total page count from the site — no hardcoded page numbers.

Usage:
    python review_scraper.py --builder shea
    python review_scraper.py --builder lennar
    python review_scraper.py --builder pulte
    python review_scraper.py --builder kb
    python review_scraper.py --slug my-builder --id 999

Output goes to builder_reviews/{slug}_reviews.csv by default.
"""

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup, NavigableString
import argparse
import math
import random
import time
import re
import os
import json
import csv
import sys
from datetime import datetime

os.environ["PYTHONUNBUFFERED"] = "1"

# ── Known builders (convenience shortcuts) ──────────────────────────────────
BUILDERS = {
    "shea":   {"slug": "shea-homes",  "id": 612},
    "lennar": {"slug": "lennar",      "id": 644},
    "pulte":  {"slug": "pulte-homes", "id": 3},
    "kb":     {"slug": "kb-home",     "id": 5},
}

# ── Scraper settings ────────────────────────────────────────────────────────
MIN_DELAY = 2
MAX_DELAY = 5
MAX_RETRIES = 3
CHECKPOINT_EVERY = 10
REVIEWS_PER_PAGE = 25  # NewHomeSource shows 25 reviews per page
CAPTCHA_TIMEOUT = 120  # seconds to wait for human to solve captcha


def log(msg):
    print(msg, flush=True)


def wait_for_human(page_handle, base_url, label="initial load"):
    """
    Poll the page until review content (h4 tags) appears.
    This gives the user time to solve a CAPTCHA / bot-check manually.
    If reviews are already on the page, returns immediately.
    """
    start = time.time()
    log(f"  [waiting] Checking for bot-check on {label}...")
    while time.time() - start < CAPTCHA_TIMEOUT:
        html = page_handle.content()
        soup = BeautifulSoup(html, "html.parser")
        # Reviews are present → page loaded successfully
        if soup.find_all("h4"):
            elapsed = time.time() - start
            if elapsed > 5:
                log(f"  [waiting] Bot-check cleared after {elapsed:.0f}s")
            return True
        # Also check for pagination links as a sign the real page loaded
        if soup.find_all("a", href=re.compile(r"/page-\d+")):
            return True
        log(f"  [waiting] No reviews yet — solve the CAPTCHA in the browser window ({int(CAPTCHA_TIMEOUT - (time.time() - start))}s remaining)")
        page_handle.wait_for_timeout(3000)

    log(f"  [timeout] Gave up waiting after {CAPTCHA_TIMEOUT}s on {label}")
    return False


# ── Page-count auto-detection ───────────────────────────────────────────────

def detect_total_pages(page_handle, base_url):
    """
    Detect total number of review pages by inspecting the first page.

    Strategies (tried in order):
      1. Parse pagination <a> links whose href contains /page-N
      2. Look for "X Reviews" text and compute ceil(X / 25)
    """
    html = page_handle.content()
    soup = BeautifulSoup(html, "html.parser")

    # Strategy 1: pagination links like /page-809
    page_links = soup.find_all("a", href=re.compile(r"/page-(\d+)"))
    if page_links:
        page_nums = []
        for a in page_links:
            m = re.search(r"/page-(\d+)", a["href"])
            if m:
                page_nums.append(int(m.group(1)))
        if page_nums:
            total = max(page_nums)
            log(f"  [auto-detect] Found pagination links → {total} pages")
            return total

    # Strategy 2: total review count text ("20216 Reviews in All Areas")
    text = soup.get_text()
    m = re.search(r"([\d,]+)\s+Reviews?\b", text)
    if m:
        total_reviews = int(m.group(1).replace(",", ""))
        total_pages = math.ceil(total_reviews / REVIEWS_PER_PAGE)
        log(f"  [auto-detect] Found {total_reviews:,} reviews → {total_pages} pages")
        return total_pages

    return None


# ── Review parsing (unchanged from shea_scraper.py) ────────────────────────

def extract_scores(h4, next_h4=None):
    """Walk forward from h4 through DOM, collect text, regex out scores."""
    text_parts = []
    el = h4.next_element
    while el:
        if el == next_h4:
            break
        if hasattr(el, "name") and el.name == "h4" and el != h4:
            break
        if isinstance(el, NavigableString) and el.strip():
            text_parts.append(el.strip())
        el = el.next_element

    full_text = " ".join(text_parts)
    scores = {}
    for label, key in [
        ("Total Score", "total_score"),
        ("Quality", "quality"),
        ("Trustworthiness", "trustworthiness"),
        ("Value", "value"),
        ("Responsiveness", "responsiveness"),
    ]:
        match = re.search(rf"{label}\s+([1-5])\b", full_text)
        scores[key] = int(match.group(1)) if match else None
    return scores


def parse_date(date_str):
    """Convert 'February 23, 2026' to '2026-02-23'."""
    try:
        dt = datetime.strptime(date_str.strip(), "%B %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return date_str


def parse_reviews(html):
    """Parse all reviews from a single page's HTML."""
    soup = BeautifulSoup(html, "html.parser")
    reviews = []
    h4_tags = soup.find_all("h4")

    for idx, h4 in enumerate(h4_tags):
        title = h4.get_text(strip=True)

        info_list = h4.find_next_sibling("ul")
        if not info_list:
            container = h4.parent
            info_list = container.find("ul") if container else None
        if not info_list:
            continue

        li_items = info_list.find_all("li")
        if len(li_items) < 3:
            continue

        reviewer_name = li_items[0].get_text(strip=True) if len(li_items) > 0 else ""
        verified = li_items[1].get_text(strip=True) if len(li_items) > 1 else ""
        date_str = li_items[2].get_text(strip=True) if len(li_items) > 2 else ""
        location = li_items[3].get_text(strip=True) if len(li_items) > 3 else ""

        if "Verified" not in verified and "HomeBuyer" not in verified:
            continue

        # extract review text
        review_text = ""
        next_sib = info_list.find_next_sibling()
        while next_sib:
            if next_sib.name == "p":
                review_text = next_sib.get_text(strip=True)
                break
            elif next_sib.name == "div":
                text = next_sib.get_text(strip=True)
                if text and text not in [
                    "Total Score", "Quality", "Trustworthiness",
                    "Value", "Responsiveness",
                ] and not re.match(r"^[1-5]$", text):
                    review_text = text
                    break
            next_sib = next_sib.find_next_sibling()

        if not review_text:
            container = h4.parent
            if container:
                all_text = container.get_text(separator="\n", strip=True)
                lines = [l.strip() for l in all_text.split("\n") if l.strip()]
                skip = {
                    title, reviewer_name, verified, date_str, location,
                    "Total Score", "Quality", "Trustworthiness", "Value",
                    "Responsiveness", "More", "... More",
                }
                review_lines = [
                    l for l in lines
                    if l not in skip and not re.match(r"^[1-5]$", l) and len(l) > 2
                ]
                if review_lines:
                    review_text = " ".join(review_lines)

        review_text = re.sub(r"\.\.\.\s*More$", "", review_text).strip()

        next_h4 = h4_tags[idx + 1] if idx + 1 < len(h4_tags) else None
        scores = extract_scores(h4, next_h4)

        reviews.append({
            "title": title,
            "reviewer_name": reviewer_name,
            "verified_homebuyer": "Yes" if "Verified" in verified else "No",
            "date": parse_date(date_str),
            "location": location,
            "review_text": review_text,
            **scores,
        })

    return reviews


# ── Checkpoint system ───────────────────────────────────────────────────────

def save_checkpoint(reviews, last_page, path):
    with open(path, "w") as f:
        json.dump({"last_page": last_page, "reviews": reviews}, f)
    log(f"  [checkpoint] Saved {len(reviews)} reviews through page {last_page}")


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        log(f"Resuming from page {data['last_page'] + 1} with {len(data['reviews'])} reviews.")
        return data["reviews"], data["last_page"]
    return [], 0


# ── Main scrape loop ───────────────────────────────────────────────────────

def scrape(slug, builder_id, output_csv, checkpoint_file, pages_override=None):
    base_url = f"https://www.newhomesource.com/builder/{slug}/reviews/{builder_id}"
    log(f"Builder: {slug} (id {builder_id})")
    log(f"Base URL: {base_url}")
    log(f"Output:   {output_csv}")

    all_reviews, last_page = load_checkpoint(checkpoint_file)
    start_page = last_page + 1
    failed_pages = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/Los_Angeles",
        )
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )

        page = context.new_page()
        page.goto(base_url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        # ── Wait for bot-check / CAPTCHA if needed ─────────────────────
        if not wait_for_human(page, base_url, "page 1"):
            log("Could not get past bot-check. Exiting.")
            browser.close()
            return []

        # ── Auto-detect total pages ─────────────────────────────────────
        if pages_override:
            total_pages = pages_override
            log(f"Using manual page count: {total_pages}")
        else:
            total_pages = detect_total_pages(page, base_url)
            if total_pages is None:
                log("ERROR: Could not auto-detect page count. Use --pages to set manually.")
                browser.close()
                return []
            log(f"Detected {total_pages} total pages")

        # Parse page 1 if we haven't yet
        if start_page == 1:
            html = page.content()
            page_reviews = parse_reviews(html)
            all_reviews.extend(page_reviews)
            log(f"Page   1/{total_pages} | {len(page_reviews):>2} reviews | Total: {len(all_reviews):>5}")
            start_page = 2

        log(f"\nScraping pages {start_page} to {total_pages}...\n")
        start_time = time.time()

        for page_num in range(start_page, total_pages + 1):
            url = f"{base_url}/page-{page_num}"
            success = False

            for attempt in range(MAX_RETRIES):
                try:
                    response = page.goto(url, wait_until="domcontentloaded", timeout=90000)

                    if response and response.status == 200:
                        page.wait_for_timeout(random.randint(1000, 2000))
                        html = page.content()
                        page_reviews = parse_reviews(html)
                        all_reviews.extend(page_reviews)
                        elapsed = time.time() - start_time
                        log(
                            f"Page {page_num:>4}/{total_pages} | "
                            f"{len(page_reviews):>2} reviews | "
                            f"Total: {len(all_reviews):>6} | "
                            f"{elapsed/60:.1f}m"
                        )
                        success = True
                        break
                    elif response and response.status == 403:
                        log(f"  [403] Page {page_num} attempt {attempt+1}/{MAX_RETRIES} — bot-check likely triggered")
                        # Re-navigate so the captcha page shows in the browser
                        page.goto(url, wait_until="domcontentloaded", timeout=90000)
                        page.wait_for_timeout(2000)
                        if wait_for_human(page, base_url, f"page {page_num} (403 retry)"):
                            # Captcha solved — retry this page
                            continue
                        # If timeout, fall through to next attempt with exponential backoff
                        wait = 30 * (2 ** attempt)
                        log(f"  [403] Falling back to {wait}s wait...")
                        page.wait_for_timeout(wait * 1000)
                    else:
                        status = response.status if response else "no response"
                        log(f"  [{status}] Page {page_num} attempt {attempt+1}/{MAX_RETRIES}")
                        page.wait_for_timeout(10000)
                except Exception as e:
                    log(f"  [ERROR] Page {page_num}: {e}")
                    # Network errors (timeout, ERR_TIMED_OUT, etc.) — connection may be stale.
                    # Re-establish by loading the base URL first, then retry the target page.
                    log(f"  [recovery] Re-establishing connection via base URL...")
                    try:
                        page.goto(base_url, wait_until="domcontentloaded", timeout=90000)
                        page.wait_for_timeout(3000)
                        wait_for_human(page, base_url, f"page {page_num} (connection recovery)")
                    except Exception:
                        pass
                    backoff = 15 * (2 ** attempt)
                    log(f"  [recovery] Waiting {backoff}s before retry...")
                    page.wait_for_timeout(backoff * 1000)

            if not success:
                failed_pages.append(page_num)
                log(f"  FAILED: Page {page_num}")

            if page_num % CHECKPOINT_EVERY == 0:
                save_checkpoint(all_reviews, page_num, checkpoint_file)

            if page_num < total_pages:
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                page.wait_for_timeout(int(delay * 1000))

        save_checkpoint(all_reviews, total_pages, checkpoint_file)
        browser.close()

    # ── Retry failed pages ──────────────────────────────────────────────
    if failed_pages:
        log(f"\nRetrying {len(failed_pages)} failed pages with fresh browser...\n")
        still_failed = []

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=False,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
            context = browser.new_context(
                viewport={"width": 1366, "height": 768},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                timezone_id="America/New_York",
            )
            context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
            )
            page = context.new_page()
            page.goto(base_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000)
            wait_for_human(page, base_url, "retry browser initial load")

            for page_num in failed_pages:
                page.wait_for_timeout(random.randint(8000, 15000))
                url = f"{base_url}/page-{page_num}"
                try:
                    response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    if response and response.status == 200:
                        page.wait_for_timeout(2000)
                        html = page.content()
                        page_reviews = parse_reviews(html)
                        all_reviews.extend(page_reviews)
                        log(f"  Recovered page {page_num} | {len(page_reviews)} reviews")
                    else:
                        still_failed.append(page_num)
                        log(f"  Still failed: page {page_num}")
                except Exception as e:
                    still_failed.append(page_num)
                    log(f"  Error page {page_num}: {e}")

            browser.close()

        failed_pages = still_failed

    # ── Deduplicate & save CSV ──────────────────────────────────────────
    log(f"\nTotal scraped: {len(all_reviews)} reviews")
    if failed_pages:
        log(f"Permanently failed pages: {failed_pages}")

    seen = set()
    unique = []
    for r in all_reviews:
        key = (r["reviewer_name"], r["date"], r["review_text"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    log(f"After dedup: {len(unique)} unique reviews (removed {len(all_reviews) - len(unique)} dupes)")

    with_scores = sum(1 for r in unique if r.get("total_score") is not None)
    log(f"Reviews with scores: {with_scores}/{len(unique)}")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    fieldnames = [
        "title", "reviewer_name", "verified_homebuyer", "date", "location",
        "review_text", "total_score", "quality", "trustworthiness",
        "value", "responsiveness",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique)

    file_size = os.path.getsize(output_csv) / 1024
    log(f"Saved to {output_csv} ({file_size:.1f} KB)")

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return unique


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape builder reviews from NewHomeSource.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python review_scraper.py --builder shea
  python review_scraper.py --builder lennar
  python review_scraper.py --builder pulte
  python review_scraper.py --builder kb
  python review_scraper.py --slug my-builder --id 999
  python review_scraper.py --builder lennar --pages 809
        """,
    )
    parser.add_argument(
        "--builder", choices=list(BUILDERS.keys()),
        help="Shorthand for a known builder (shea, lennar, pulte, kb)",
    )
    parser.add_argument("--slug", help="Builder URL slug (e.g. 'shea-homes')")
    parser.add_argument("--id", type=int, help="Builder ID in the NewHomeSource URL")
    parser.add_argument("--output", help="Output CSV path (default: builder_reviews/{slug}_reviews.csv)")
    parser.add_argument("--pages", type=int, help="Override auto-detected page count")

    args = parser.parse_args()

    if args.builder:
        info = BUILDERS[args.builder]
        slug = info["slug"]
        builder_id = info["id"]
    elif args.slug and args.id:
        slug = args.slug
        builder_id = args.id
    else:
        parser.error("Provide --builder OR both --slug and --id")
        return

    output_csv = args.output or os.path.join("builder_reviews", f"{slug}_reviews.csv")
    checkpoint_file = os.path.join("builder_reviews", f".{slug}_checkpoint.json")

    scrape(slug, builder_id, output_csv, checkpoint_file, pages_override=args.pages)


if __name__ == "__main__":
    main()
