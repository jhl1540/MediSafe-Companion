import re
import time
import sys
import argparse
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

BASE = "https://ddinter.scbdd.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/119.0 Safari/537.36"
}

DETAIL_HREF_RE = re.compile(r"^/ddinter/interact/\d+/?$")

def fetch(url, session, **kwargs):
    r = session.get(url, headers=HEADERS, timeout=30, **kwargs)
    r.raise_for_status()
    return r

def textnorm(s):
    return re.sub(r"\s+", " ", s or "").strip()

def extract_detail_links_from_drug_page(drug_id, session):
    """Plan A: static parse of the drug detail page to collect all /ddinter/interact/<id>/ links."""
    url = f"{BASE}/ddinter/drug-detail/{drug_id}/"
    res = fetch(url, session)
    soup = BeautifulSoup(res.text, "lxml")

    # Collect any anchors that look like detail links in the page
    hrefs = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if DETAIL_HREF_RE.match(href):
            hrefs.add(urljoin(BASE, href))

    # Sometimes pagination exists; collect detail links across pages if page links found
    page_candidates = []
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h and "drug-detail" in h and drug_id in h:
            page_candidates.append(urljoin(BASE, h))
    # De-dup; include current page first
    page_candidates = [url] + [p for p in page_candidates if p != url]
    page_candidates = list(dict.fromkeys(page_candidates))  # stable unique

    for p in page_candidates:
        try:
            resp = fetch(p, session)
            sp = BeautifulSoup(resp.text, "lxml")
            for a in sp.find_all("a", href=True):
                href = a["href"].strip()
                if DETAIL_HREF_RE.match(href):
                    hrefs.add(urljoin(BASE, href))
        except Exception:
            continue

    return sorted(hrefs)

def parse_interact_page(detail_url, session):
    """Parse a /ddinter/interact/<id>/ page into a structured dict by label splitting."""
    html = fetch(detail_url, session).text
    soup = BeautifulSoup(html, "lxml")

    # A robust strategy: build a plain text with preserved order of visible strings,
    # then split by our known section labels.
    # Known labels seen on site: "ID", "Interaction", "Management", "References",
    # and "Alternative for <DrugName>" (twice, one per drug).
    text = " ".join(soup.stripped_strings)
    text = re.sub(r"\s+", " ", text)

    # Extract Severity + Mechanism from the heading line if present
    # e.g., "Interaction between Fluconazole and Ethotoin Major Metabolism"
    mhead = re.search(r"Interaction between (.+?) and (.+?)\s+(\w+)\s+(\w+)", text)
    drug1_name = drug2_name = severity = mechanism = ""
    if mhead:
        drug1_name = mhead.group(1).strip()
        drug2_name = mhead.group(2).strip()
        severity = mhead.group(3).strip()
        mechanism = mhead.group(4).strip()

    # ID line: "ID DDInter743 and DDInter696"
    m_id = re.search(r"\bID\s+(DDInter\d+)\s+and\s+(DDInter\d+)\b", text)
    drug1_id = m_id.group(1) if m_id else ""
    drug2_id = m_id.group(2) if m_id else ""

    # Interaction
    interaction = ""
    m_inter = re.search(r"\bInteraction\s+(.*?)\s+Management\b", text)
    if m_inter:
        interaction = m_inter.group(1).strip()
    else:
        # Fallback if Management not present for some reason
        m_inter2 = re.search(r"\bInteraction\s+(.*?)(?:References|Alternative for|$)", text)
        if m_inter2:
            interaction = m_inter2.group(1).strip()

    # Management
    management = ""
    m_mgmt = re.search(r"\bManagement\s+(.*?)\s+(?:References|Alternative for|$)", text)
    if m_mgmt:
        management = m_mgmt.group(1).strip()

    # References: try to capture between 'References' and next 'Alternative for' or end
    references = ""
    m_refs = re.search(r"\bReferences\s+(.*?)\s+(?:Alternative for|$)", text)
    if m_refs:
        references = m_refs.group(1).strip()

    # Alternatives: there are usually two blocks "Alternative for <Drug1>" and "... for <Drug2>"
    # We'll greedily capture names until the next "Alternative for" or end.
    alt_for_1_name, alt_for_1, alt_for_2_name, alt_for_2 = "", "", "", ""

    # Find all "Alternative for <Name>" markers with positions
    alt_iter = list(re.finditer(r"Alternative for\s+([^ ]+)(.*?)((?=Alternative for\s+)|$)", text))
    # Sometimes the drug name after "Alternative for" can be multi-word (e.g., "St. John's Wort").
    # Try a more generous capture by looking ahead for an ATC code or double space separation.
    if not alt_iter:
        alt_iter = list(re.finditer(r"Alternative for\s+(.+?)\s{2,}(.*?)(?=(?:Alternative for\s+)|$)", text))

    # If exactly two, map to drug1/drug2 by string similarity to names when possible
    def parse_alt_block(name, block):
        # The block contains a mess of codes (ATC) and drug names separated by spaces.
        # Extract capitalized multi-words by splitting on two+ spaces or bullets.
        # Simpler approach: split by two spaces, then keep tokens that look like words with letters.
        # Also remove "More".
        raw = block
        raw = raw.replace("More", " ")
        # keep comma-separated if any
        items = re.split(r"\s{2,}|\s{1,}\u00B7\s{1,}| {1,}\u2022 {1,}| ; | , ", raw)
        # light clean & filter short tokens
        cleaned = []
        for it in items:
            s = textnorm(it)
            if not s:
                continue
            # Skip pure codes like "J01R" (keep them? You can keep by toggling the condition)
            # We'll keep names and drop 3-5 char all caps codes:
            if re.fullmatch(r"[A-Z0-9]{3,5}", s):
                continue
            if s.lower().startswith("alternative for"):
                continue
            cleaned.append(s)
        # De-dup keeping order
        seen = set()
        out = []
        for c in cleaned:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return name, "; ".join(out[:200])  # keep it bounded

    if alt_iter:
        # Build list of (name, block)
        blocks = []
        for m in alt_iter:
            name = textnorm(m.group(1))
            block = textnorm(m.group(2))
            blocks.append((name, block))

        # If there are more than two blocks for any reason, take first two
        if len(blocks) >= 1:
            alt_for_1_name, alt_for_1 = parse_alt_block(blocks[0][0], blocks[0][1])
        if len(blocks) >= 2:
            alt_for_2_name, alt_for_2 = parse_alt_block(blocks[1][0], blocks[1][1])

    # Detail numeric id from URL
    murl = re.search(r"/interact/(\d+)/", detail_url)
    pair_id = murl.group(1) if murl else ""

    return {
        "pair_id": pair_id,
        "detail_url": detail_url,
        "severity": severity,
        "mechanism": mechanism,
        "drug1_id": drug1_id,
        "drug1_name": drug1_name,
        "drug2_id": drug2_id,
        "drug2_name": drug2_name,
        "interaction": interaction,
        "management": management,
        "references": references,
        "alternative_for_drug1_name": alt_for_1_name,
        "alternatives_for_drug1": alt_for_1,
        "alternative_for_drug2_name": alt_for_2_name,
        "alternatives_for_drug2": alt_for_2,
    }

def try_collect_links_with_selenium(drug_id):
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        return []

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1600,1200")
    options.add_argument("--ignore-certificate-errors")  # <-- accept expired certs
    options.set_capability("acceptInsecureCerts", True)   # Selenium 4 way

    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

    try:
        url = f"{BASE}/ddinter/drug-detail/{drug_id}/"
        driver.get(url)

        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, "a")))

        all_links = set()
        def grab_links_on_page():
            anchors = driver.find_elements(By.TAG_NAME, "a")
            for a in anchors:
                href = a.get_attribute("href") or ""
                if re.search(r"/ddinter/interact/\d+/?$", href):
                    all_links.add(href)

        grab_links_on_page()

        # Optional: paginate via “Next”
        for _ in range(200):
            next_btns = driver.find_elements(By.XPATH, "//a[contains(., 'Next')]")
            if not next_btns:
                break
            next_btns[0].click()
            time.sleep(1.2)
            grab_links_on_page()

        return sorted(all_links)
    finally:
        driver.quit()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug-id", required=True, help="e.g., DDInter14 for Acetaminophen")
    ap.add_argument("--out", default="ddinter_interactions.csv")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"[1/3] Scanning interaction links for {args.drug_id} ...")
    links = extract_detail_links_from_drug_page(args.drug_id, session)
    if not links:
        print("No detail links found via requests. Falling back to Selenium...")
        links = try_collect_links_with_selenium(args.drug_id)

    if not links:
        print("Could not find any interaction detail links. "
              "The page may be protected or changed its structure.")
        sys.exit(1)

    print(f"Found {len(links)} detail pages. Fetching & parsing…")

    rows = []
    for url in tqdm(links, ncols=88):
        try:
            rows.append(parse_interact_page(url, session))
            time.sleep(0.1)  # be gentle
        except Exception as e:
            # Keep going; record minimal info
            rows.append({
                "pair_id": "",
                "detail_url": url,
                "severity": "",
                "mechanism": "",
                "drug1_id": "",
                "drug1_name": "",
                "drug2_id": "",
                "drug2_name": "",
                "interaction": f"ERROR: {e}",
                "management": "",
                "references": "",
                "alternative_for_drug1_name": "",
                "alternatives_for_drug1": "",
                "alternative_for_drug2_name": "",
                "alternatives_for_drug2": "",
            })

    print(f"[3/3] Writing CSV -> {args.out}")
    df = pd.DataFrame(rows)
    # Optional: sort by pair_id numeric if present
    with pd.option_context('mode.chained_assignment', None):
        df["pair_id_num"] = pd.to_numeric(df["pair_id"], errors="coerce")
        df.sort_values(["pair_id_num", "drug2_name"], inplace=True)
        df.drop(columns=["pair_id_num"], inplace=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print("Done.")

if __name__ == "__main__":
    main()
