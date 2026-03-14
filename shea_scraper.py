"""
Shea Homes Review Scraper
Run directly: python shea_scraper.py

Fixes from v1:
  - Scores now parsed correctly (walks forward through DOM, regex extraction)
  - Dates output as YYYY-MM-DD so Excel doesn't show ####
"""

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup, NavigableString
import random
import time
import re
import os
import json
import csv
import sys
from datetime import datetime

# force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

# --- CONFIG ---
BASE_URL = "https://www.newhomesource.com/builder/shea-homes/reviews/612"
TOTAL_PAGES = 82
MIN_DELAY = 2
MAX_DELAY = 5
MAX_RETRIES = 3
CHECKPOINT_EVERY = 10
CHECKPOINT_FILE = "scrape_checkpoint.json"
OUTPUT_CSV = "shea_homes_reviews.csv"


def log(msg):
    print(msg, flush=True)


def extract_scores(h4, next_h4=None):
    """
    Walk forward from h4 through all DOM elements until the next h4,
    collect raw text, then regex out score values.
    Works regardless of how the HTML is nested.
    """
    text_parts = []
    el = h4.next_element
    while el:
        # stop at next review
        if el == next_h4:
            break
        if hasattr(el, "name") and el.name == "h4" and el != h4:
            break
        # collect raw text nodes
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
        if match:
            scores[key] = int(match.group(1))
        else:
            scores[key] = None

    return scores


def parse_date(date_str):
    """Convert 'February 23, 2026' to '2026-02-23' for Excel compatibility."""
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

        # find the ul with reviewer metadata
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

        # extract review text - walk siblings after the ul
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

        # fallback: grab text from parent, excluding metadata and scores
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

        # extract scores by walking forward through DOM
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


def save_checkpoint(reviews, last_page):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_page": last_page, "reviews": reviews}, f)
    log(f"  [checkpoint] Saved {len(reviews)} reviews through page {last_page}")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        log(f"Resuming from page {data['last_page'] + 1} with {len(data['reviews'])} reviews already collected.")
        return data["reviews"], data["last_page"]
    return [], 0


def scrape():
    all_reviews, last_page = load_checkpoint()
    start_page = last_page + 1
    failed_pages = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,  # change to False to see browser / solve CAPTCHAs
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

        log("Warming up browser session...")
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        log(f"Scraping pages {start_page} to {TOTAL_PAGES}...\n")
        start_time = time.time()

        for page_num in range(start_page, TOTAL_PAGES + 1):
            url = BASE_URL if page_num == 1 else f"{BASE_URL}/page-{page_num}"
            success = False

            for attempt in range(MAX_RETRIES):
                try:
                    response = page.goto(
                        url, wait_until="domcontentloaded", timeout=60000
                    )

                    if response and response.status == 200:
                        page.wait_for_timeout(random.randint(1000, 2000))
                        html = page.content()
                        page_reviews = parse_reviews(html)
                        all_reviews.extend(page_reviews)
                        elapsed = time.time() - start_time
                        log(
                            f"Page {page_num:>3}/{TOTAL_PAGES} | "
                            f"{len(page_reviews):>2} reviews | "
                            f"Total: {len(all_reviews):>5} | "
                            f"{elapsed/60:.1f}m"
                        )
                        success = True
                        break

                    elif response and response.status == 403:
                        wait = 30 * (2 ** attempt)
                        log(
                            f"  [403] Page {page_num} attempt "
                            f"{attempt+1}/{MAX_RETRIES}. Waiting {wait}s..."
                        )
                        page.wait_for_timeout(wait * 1000)

                    else:
                        status = response.status if response else "no response"
                        log(
                            f"  [{status}] Page {page_num} attempt "
                            f"{attempt+1}/{MAX_RETRIES}"
                        )
                        page.wait_for_timeout(10000)

                except Exception as e:
                    log(f"  [ERROR] Page {page_num}: {e}")
                    page.wait_for_timeout(15000)

            if not success:
                failed_pages.append(page_num)
                log(f"  FAILED: Page {page_num}")

            if page_num % CHECKPOINT_EVERY == 0:
                save_checkpoint(all_reviews, page_num)

            if page_num < TOTAL_PAGES:
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                page.wait_for_timeout(int(delay * 1000))

        save_checkpoint(all_reviews, TOTAL_PAGES)
        browser.close()

    # --- RETRY FAILED PAGES ---
    if failed_pages:
        log(f"\nRetrying {len(failed_pages)} failed pages with fresh browser...\n")
        still_failed = []

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
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

            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(5000)

            for page_num in failed_pages:
                page.wait_for_timeout(random.randint(8000, 15000))
                url = BASE_URL if page_num == 1 else f"{BASE_URL}/page-{page_num}"

                try:
                    response = page.goto(
                        url, wait_until="domcontentloaded", timeout=60000
                    )
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

    # --- SAVE CSV ---
    elapsed = time.time() - start_time
    log(f"\nTotal scraped: {len(all_reviews)} reviews in {elapsed/60:.1f} minutes.")
    if failed_pages:
        log(f"Permanently failed pages: {failed_pages}")

    # deduplicate
    seen = set()
    unique = []
    for r in all_reviews:
        key = (r["reviewer_name"], r["date"], r["review_text"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    log(f"After dedup: {len(unique)} unique reviews (removed {len(all_reviews) - len(unique)} dupes)")

    # count how many have scores
    with_scores = sum(1 for r in unique if r.get("total_score") is not None)
    log(f"Reviews with scores: {with_scores}/{len(unique)}")

    fieldnames = [
        "title", "reviewer_name", "verified_homebuyer", "date", "location",
        "review_text", "total_score", "quality", "trustworthiness",
        "value", "responsiveness",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique)

    file_size = os.path.getsize(OUTPUT_CSV) / 1024
    log(f"Saved to {OUTPUT_CSV} ({file_size:.1f} KB)")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    log("Done! Ready for sentiment analysis.")
    return unique


if __name__ == "__main__":
    scrape()
