import re
import time
import argparse
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import re, time

# --- Selenium setup (only) ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

BASE = "https://ddinter.scbdd.com"

def make_driver(headless=True):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1600,1200")
    # Accept the expired cert:
    opts.add_argument("--ignore-certificate-errors")
    opts.set_capability("acceptInsecureCerts", True)
    # Be a little nicer
    opts.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    return driver

DETAIL_RE = re.compile(r"/ddinter/interact/(\d+)/?$")

def _click_interactions_tab_if_present(driver):
    """Some DDInter drug pages have multiple tabs; ensure we're on 'Interactions'."""
    try:
        # Try by link text
        tabs = driver.find_elements(By.XPATH, "//a[contains(., 'Interaction') or contains(., 'Interactions')]")
        for t in tabs:
            if t.is_displayed():
                driver.execute_script("arguments[0].click();", t)
                time.sleep(0.6)
                break
    except Exception:
        pass

def _try_set_rows_per_page_to_100(driver):
    """Try to bump the table page length to 100 for common table libs."""
    try:
        # DataTables length <select>
        selects = driver.find_elements(By.CSS_SELECTOR, ".dataTables_length select, select[name$='_length'], div.layui-laypage select, .ant-select-selector, .el-select")
        # If it's a real <select>, use Select API
        real_selects = [s for s in selects if s.tag_name.lower() == "select"]
        if real_selects:
            sel = Select(real_selects[0])
            # Pick the largest numeric option available
            numeric_options = []
            for o in sel.options:
                val = (o.get_attribute("value") or o.text or "").strip()
                try:
                    numeric_options.append((int(val), val))
                except:
                    try:
                        numeric_options.append((int(o.text.strip()), o.text.strip()))
                    except:
                        pass
            if numeric_options:
                max_val = str(max(numeric_options)[0])
                sel.select_by_value(max_val)
                time.sleep(0.8)
                return True

        # If it’s not a native <select>, try a JS change (works on some UIs)
        driver.execute_script("""
          (function(){
            var picks = Array.from(document.querySelectorAll('select'));
            for (const s of picks) {
              var has = Array.from(s.options).map(o => o.textContent.trim());
              if (has.includes('100') || has.includes('50')) {
                s.value = has.includes('100') ? '100' : '50';
                s.dispatchEvent(new Event('change', {bubbles:true}));
                return;
              }
            }
          })();
        """)
        time.sleep(0.8)
        return True
    except Exception:
        return False

def _find_next_button(driver):
    """Return a visible, enabled 'next' element from common table pagers."""
    candidates = []
    selectors = [
        # DataTables
        "a.paginate_button.next",
        "li.paginate_button.next > a, li.paginate_button.next > span, li.next > a, li.next > button",
        # Generic text/button
        "//a[contains(., 'Next') or contains(., 'next')]",
        "//button[contains(., 'Next') or contains(., 'next')]",
        "//a[@rel='next']",
        "//button[@rel='next']",
        "//li[contains(@class,'next')]//a|//li[contains(@class,'next')]//button",
        # Ant Design
        ".ant-pagination-next button, .ant-pagination-next a",
        # Element UI
        ".el-pagination__next, .btn-next",
        # layui
        "a.layui-laypage-next",
        # Bootstrap variants
        "ul.pagination li.next a, ul.pagination li.next button",
    ]

    # Try CSS selectors first
    for css in [s for s in selectors if not s.startswith("//")]:
        for el in driver.find_elements(By.CSS_SELECTOR, css):
            candidates.append(el)
    # Then XPaths
    for xp in [s for s in selectors if s.startswith("//")]:
        for el in driver.find_elements(By.XPATH, xp):
            candidates.append(el)

    # Filter visible/enabled and not disabled by class or aria
    filtered = []
    for el in candidates:
        try:
            cls = (el.get_attribute("class") or "").lower()
            aria = (el.get_attribute("aria-disabled") or "").lower()
            if not el.is_displayed():
                continue
            if "disabled" in cls or aria == "true":
                continue
            filtered.append(el)
        except Exception:
            continue

    # Return the first if any
    return filtered[0] if filtered else None

def get_all_detail_links(driver, drug_id: str, wait_timeout=25):
    """Collect all /ddinter/interact/<id>/ links across every page."""
    url = f"https://ddinter.scbdd.com/ddinter/drug-detail/{drug_id}/"
    driver.get(url)
    wait = WebDriverWait(driver, wait_timeout)

    # Accept expired TLS cert is handled by your Options (acceptInsecureCerts)
    # Ensure content is loaded
    wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, "body")))
    _click_interactions_tab_if_present(driver)
    time.sleep(0.6)

    # Sometimes rows are lazy — give a small scroll pass
    for y in range(0, 1500, 300):
        driver.execute_script(f"window.scrollTo(0, {y});")
        time.sleep(0.15)

    # Try to show more rows per page
    _try_set_rows_per_page_to_100(driver)

    links = set()

    def collect_links_on_page():
        anchors = driver.find_elements(By.TAG_NAME, "a")
        new_count = 0
        for a in anchors:
            href = a.get_attribute("href") or ""
            if DETAIL_RE.search(href):
                if href not in links:
                    links.add(href)
                    new_count += 1
        return new_count

    # First page
    collect_links_on_page()

    # Keep clicking 'Next' until none
    pages = 1
    while True:
        next_btn = _find_next_button(driver)
        if not next_btn:
            break

        # Remember an element from the current page so we can wait for staleness
        sentinel = None
        try:
            sentinel = driver.find_elements(By.CSS_SELECTOR, "a[href*='/ddinter/interact/']")[0]
        except Exception:
            pass

        # Scroll next button into view and click via JS (most reliable)
        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", next_btn)
            time.sleep(0.1)
            driver.execute_script("arguments[0].click();", next_btn)
        except Exception:
            # Fallback to .click()
            try:
                next_btn.click()
            except Exception:
                break

        # Wait for page content to change (staleness of a known link or just small sleep)
        if sentinel:
            try:
                WebDriverWait(driver, 10).until(EC.staleness_of(sentinel))
            except Exception:
                time.sleep(0.8)
        else:
            time.sleep(0.8)

        # Give the table a moment to redraw
        time.sleep(0.4)
        pages += 1
        added = collect_links_on_page()

        # Optional: stop if clicking next didn’t change anything for N tries
        if added == 0:
            # Still advance if the paginator is moving; otherwise break after a few no-op pages
            pass

        # Safety cap: don’t loop forever
        if pages > 1000:
            break

    return sorted(links)

def parse_detail_with_selenium(driver, detail_url: str, wait_timeout=20):
    """Open a detail page in Selenium, parse with BeautifulSoup, and return a dict of fields."""
    driver.get(detail_url)
    WebDriverWait(driver, wait_timeout).until(
        EC.presence_of_all_elements_located((By.TAG_NAME, "body"))
    )
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
    text = " ".join(soup.stripped_strings)
    text = re.sub(r"\s+", " ", text)

    # IDs: "ID DDInter14 and DDInterXYZ"
    drug1_id = drug2_id = ""
    m_id = re.search(r"\bID\s+(DDInter\d+)\s+and\s+(DDInter\d+)\b", text)
    if m_id:
        drug1_id, drug2_id = m_id.group(1), m_id.group(2)

    # Interaction
    interaction = ""
    m_inter = re.search(r"\bInteraction\s+(.*?)\s+Management\b", text)
    if not m_inter:
        m_inter = re.search(r"\bInteraction\s+(.*?)(?:References|Alternative for|$)", text)
    if m_inter:
        interaction = m_inter.group(1).strip()

    # Management
    management = ""
    m_mgmt = re.search(r"\bManagement\s+(.*?)\s+(?:References|Alternative for|$)", text)
    if m_mgmt:
        management = m_mgmt.group(1).strip()

    # References
    references = ""
    m_refs = re.search(r"\bReferences\s+(.*?)\s+(?:Alternative for|$)", text)
    if m_refs:
        references = m_refs.group(1).strip()

    # Alternatives — capture each "Alternative for <DrugName> <list...>"
    alt_blocks = []
    # Try two patterns: (1) generic lookahead to next "Alternative for" or end
    for m in re.finditer(r"Alternative for\s+(.+?)\s+(.*?)(?=Alternative for\s+|$)", text):
        alt_blocks.append((m.group(1).strip(), m.group(2).strip()))

    # Clean the alternatives list a bit
    def clean_alts(block_text):
        raw = block_text.replace("More", " ")
        parts = re.split(r"\s{2,}| {1,}[•·] {1,}| ; | , ", raw)
        out = []
        seen = set()
        for p in parts:
            s = re.sub(r"\s+", " ", p).strip()
            if not s:
                continue
            # skip short all-caps codes like ATC codes if you don’t want them:
            if re.fullmatch(r"[A-Z0-9]{3,6}", s):
                continue
            if s.lower().startswith("alternative for"):
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
        return "; ".join(out[:200])

    # Expect two blocks: one for base drug (Acetaminophen) and one for the counterpart drug.
    alt_for_acetaminophen = ""
    alt_for_other_drug = ""
    other_drug_name = ""

    # Identify which block is for Acetaminophen specifically; the other becomes "other".
    for (name, blk) in alt_blocks:
        cleaned = clean_alts(blk)
        if re.search(r"\bAcetaminophen\b", name, re.IGNORECASE):
            alt_for_acetaminophen = cleaned
        else:
            if not other_drug_name:
                other_drug_name = name
                alt_for_other_drug = cleaned

    # Extract numeric pair id from URL
    murl = DETAIL_RE.search(detail_url)
    pair_id = murl.group(1) if murl else ""

    return {
        "pair_id": pair_id,
        "detail_url": detail_url,
        "drug1_id": drug1_id,
        "drug2_id": drug2_id,
        "interaction": interaction,
        "management": management,
        "references": references,
        "alternative_for_acetaminophen": alt_for_acetaminophen,
        "other_drug_name": other_drug_name,
        "alternative_for_other_drug": alt_for_other_drug,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug-id", required=True, help="e.g., DDInter14")
    ap.add_argument("--out", default="ddinter_interactions.csv")
    ap.add_argument("--no-headless", action="store_true", help="Run browser with UI (debug).")
    args = ap.parse_args()

    driver = make_driver(headless=not args.no_headless)
    try:
        print(f"Collecting detail links for {args.drug_id} ...")
        links = get_all_detail_links(driver, args.drug_id)
        if not links:
            print("No detail links found. If the table is lazy-loaded, scroll a bit and try again with --no-headless.")
            return

        print(f"Found {len(links)} detail pages. Parsing…")
        rows = []
        for url in tqdm(links, ncols=88):
            try:
                rows.append(parse_detail_with_selenium(driver, url))
                time.sleep(0.25)  # be gentle
            except Exception as e:
                rows.append({
                    "pair_id": "",
                    "detail_url": url,
                    "drug1_id": "",
                    "drug2_id": "",
                    "interaction": f"ERROR: {e}",
                    "management": "",
                    "references": "",
                    "alternative_for_acetaminophen": "",
                    "other_drug_name": "",
                    "alternative_for_other_drug": "",
                })

        print(f"Writing CSV -> {args.out}")
        df = pd.DataFrame(rows)
        df.to_csv(args.out, index=False, encoding="utf-8")
        print("Done.")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()


# python ddinter_scrape_selenium_only.py --drug-id DDInter263 --out caffeine_ddis.csv