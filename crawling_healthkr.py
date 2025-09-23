import time
import re
import csv
import sys
from typing import Dict, List, Tuple, Optional
import requests
from html import unescape
from bs4 import BeautifulSoup, NavigableString, Tag

BASE = "https://www.health.kr/searchDrug"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; data-collect/1.0; +https://example.org/bot)",
    "Accept-Language": "ko,en;q=0.9",
}

DETAIL_URL = BASE + "/result_drug.asp?drug_cd={drug_cd}"
INTERACT_URL = BASE + "/result_interaction.asp?drug_cd={drug_cd}"

# def fetch(url: str, timeout: int = 20) -> BeautifulSoup:
#     """GET the URL and return BeautifulSoup, with safe encoding handling."""
#     r = requests.get(url, headers=HEADERS, timeout=timeout)
#     # Try to respect server-declared encoding; if missing, fallback to apparent
#     if r.encoding is None or r.encoding.lower() in ("iso-8859-1", "us-ascii"):
#         r.encoding = r.apparent_encoding or "utf-8"
#     return BeautifulSoup(r.text, "html.parser")
  
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; data-collect/1.0)",
    "Accept-Language": "ko,en;q=0.9",
    "Referer": "https://www.health.kr/main.asp",
})

def fetch(url, timeout=20):
    r = SESSION.get(url, timeout=timeout)
    enc = (r.encoding or "").lower()
    if not enc or enc in ("iso-8859-1", "us-ascii"):
        r.encoding = r.apparent_encoding or "utf-8"

    html = r.text.replace("\r\n", "\n")
    for parser in ("html5lib", "lxml", "html.parser"):
        try:
            return BeautifulSoup(html, parser)
        except Exception:
            continue
    return BeautifulSoup(html, "html.parser")
  

def text_of(node: Optional[Tag]) -> str:
    """Flatten and clean text of a node."""
    if node is None:
        return ""
    txt = " ".join(node.stripped_strings)
    return re.sub(r"\s+", " ", txt).strip()
  

def _valid_clip_value(val: str) -> bool:
    if not val:
        return False
    v = val.strip()
    if v.lower() in ("all", "전체"):
        return False
    return len(v) >= 4

def _scan_clipboard_payloads(soup: BeautifulSoup, start_node: Tag, search_limit: int) -> str:
    """Scan forward for clipboard text in three ways:
       (1) data-clipboard-text
       (2) data-clipboard-target -> resolve CSS selector and read its value/text
       (3) <input>/<textarea> value
    """
    import re
    best = ""
    steps = 0
    for el in start_node.next_elements:
        if steps > search_limit:
            break
        steps += 1
        if not isinstance(el, Tag):
            continue

        # (1) direct payload
        if el.has_attr("data-clipboard-text"):
            v = el.get("data-clipboard-text", "").strip()
            if _valid_clip_value(v): return v
            if v and len(v) > len(best): best = v

        # (2) target indirection
        target = el.get("data-clipboard-target")
        if target:
            try:
                tgt = soup.select_one(target)
                if tgt:
                    v = (tgt.get("value") or tgt.get_text(strip=True) or "").strip()
                    if _valid_clip_value(v): return v
                    if v and len(v) > len(best): best = v
            except Exception:
                pass

        # (3) raw inputs
        if el.name in ("input", "textarea"):
            v = (el.get("value") or "").strip()
            if _valid_clip_value(v): return v
            if v and len(v) > len(best): best = v

        # Stop at next big heading
        if re.match(r"^h[1-6]$", el.name) and steps > 6:
            break

    return best if _valid_clip_value(best) else ""

def get_clipboard_text_near_label(soup: BeautifulSoup, label_regex: str, search_limit: int = 200) -> str:
    label = soup.find(string=re.compile(label_regex))
    if not label:
        return ""
    start = label.parent if hasattr(label, "parent") else soup
    return _scan_clipboard_payloads(soup, start, search_limit)

def get_clipboard_text_in_section(soup: BeautifulSoup, section_title_regex: str, search_limit: int = 600) -> str:
    head = soup.find(string=re.compile(section_title_regex))
    if not head:
        return ""
    start = head.parent if hasattr(head, "parent") else soup
    return _scan_clipboard_payloads(soup, start, search_limit)

def extract_product_name(soup: BeautifulSoup) -> str:
    # Primary: clipboard right after "제품명"
    val = get_clipboard_text_near_label(soup, r"^\s*제품명\s*$")
    if _valid_clip_value(val):
        return val

    # Secondary: read the same row (e.g., <tr><th>제품명</th><td>...</td></tr>)
    label = soup.find(string=re.compile(r"^\s*제품명\s*$"))
    if label:
        tr = label.find_parent("tr")
        if tr:
            tds = tr.find_all("td")
            if tds:
                txt = " ".join(tds[-1].stripped_strings)
                txt = txt.replace("복사", "").strip()
                if _valid_clip_value(txt):
                    return txt

    # Fallback: packaging table
    pkg_hdr = soup.find(string=re.compile(r"심평원\s*기준\s*포장단위"))
    if pkg_hdr:
        tbl = pkg_hdr.find_parent()
        if tbl:
            txt = " ".join(tbl.stripped_strings)
            m = re.search(r"([가-힣A-Za-z0-9\.\-]+(?:정|캡슐|현탁액|시럽|연질캡슐)?[^\s]*\s*[0-9]+(?:mg|밀리그람))|([A-Za-z].*?Tab\.\s*[0-9]+mg)", txt)
            if m:
                name = (m.group(1) or m.group(2)).strip()
                name = re.sub(r"([0-9]+)\s*밀리그람", r"\1mg", name)
                return name

    return ""

def extract_ingredients(soup: BeautifulSoup) -> str:
    # Primary: clipboard in "성분 / 함량" section
    val = get_clipboard_text_in_section(soup, r"성분\s*/\s*함량")
    if _valid_clip_value(val):
        return val.replace("㎎", "mg").replace(" 밀리그람", "mg").strip()

    # Secondary: visible text under the same section (regex around 'mg')
    head = soup.find(string=re.compile(r"성분\s*/\s*함량"))
    if head:
        cont = head.parent
        if cont:
            txt = " ".join(cont.get_text("\n", strip=True).split())
            m = re.search(r"([A-Za-z][A-Za-z\- ]+)\s*([가-힣\(\)·\s]+)\s*([0-9]+)\s*(?:mg|㎎|밀리그람)", txt)
            if m:
                en, ko, mg = m.group(1).strip(), m.group(2).strip(), m.group(3) + "mg"
                return f"{en} {ko} {mg}".replace("  ", " ").strip()

    return ""

def get_following_section_text(soup: BeautifulSoup, heading_regex: str) -> str:
  
    """
    Find a heading that matches heading_regex (e.g., '효능.*효과' or '용법.*용량')
    and return all text until the next heading of similar level.
    Works on this site where those sections are rendered as blocks under '허가정보 ∙ 복약정보'.
    """
    # Try h3/h4/h5 headings
    for tag in soup.find_all(re.compile(r"^h[2-5]$")):
        if re.search(heading_regex, text_of(tag), flags=re.I):
            # Collect siblings until next heading
            texts = []
            for sib in tag.find_all_next():
                if sib == tag:
                    continue
                if sib.name and re.match(r"^h[2-5]$", sib.name):
                    break
                # The site places content in <p>, <li>, or raw text
                if isinstance(sib, Tag) and sib.name in ("p", "div", "li", "ul", "ol", "br"):
                    t = text_of(sib)
                    if t:
                        texts.append(t)
                # If we run into the footer/legal, bail out
                if "법적 고지" in text_of(sib):
                    break
            # Post-process: keep first ~10 lines to avoid trailing legal repeats
            combined = "\n".join(texts).strip()
            # Trim duplicated “주효능·효과” lines if repeated in footer snapshots
            combined = re.sub(r"(법적 고지.*)$", "", combined, flags=re.S)
            return combined.strip()
    return ""
  
def get_section_text_between(soup: BeautifulSoup, start_regex: str, stop_regexes: List[str]) -> str:
    import re
    # Find an actual heading element whose FULL TEXT matches the pattern
    heading = soup.find(
        lambda t: isinstance(t, Tag)
        and t.name in ("h2", "h3", "h4", "h5")
        and re.search(start_regex, t.get_text(" ", strip=True))
    )
    if not heading:
        return ""

    def is_stop(txt: str) -> bool:
        for pat in stop_regexes:
            if re.search(pat, txt):
                return True
        return False

    parts = []
    # Walk **siblings** after the heading, stop at the next heading or a stop text
    sib = heading.next_sibling
    while sib:
        if isinstance(sib, Tag):
            if sib.name in ("h2", "h3", "h4", "h5"):
                break
            txt = " ".join(sib.stripped_strings).strip()
            if txt:
                if is_stop(txt):
                    break
                # Only keep reasonable blocks
                if sib.name in ("p", "li", "div") and txt != "복사":
                    parts.append(txt)
        sib = sib.next_sibling

    out = " ".join(parts)
    out = re.sub(r"\s*•\s*", " · ", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out
  
def parse_detail_page(drug_cd: str) -> Dict[str, str]:
    url = DETAIL_URL.format(drug_cd=drug_cd)
    soup = fetch(url)
    data = {
        "drug_cd": drug_cd,
        "제품명": extract_product_name(soup),
        "성분": extract_ingredients(soup),
        "효능·효과": get_section_text_between(
            soup,
            r"효능\s*·\s*효과",
            [r"용법\s*·\s*용량", r"사용상", r"주의사항", r"보관", r"저장", r"상호작용"]
        ),
        "용법·용량": get_section_text_between(
            soup,
            r"용법\s*·\s*용량",
            [r"효능\s*·\s*효과", r"사용상", r"주의사항", r"보관", r"저장", r"상호작용"]
        ),
        "source_url": url,
    }
    return {k: (v or "").strip() for k, v in data.items()}

def parse_interaction_rows(soup):
    """
    DOM-based parser:
      - Find every exact '해당제품' marker node
      - Pair markers [0,1], [2,3], ... as (성분1, 성분2)
      - 성분명 = nearest preceding <a> (or text) *bounded by a stop node*
      - 내용 = unique text from <li>/<p> between pair's 2nd marker and next pair's 1st marker
    """
    import re
    from bs4 import Tag, NavigableString

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.replace("\xa0", " ").strip())

    def node_text(n) -> str:
        if n is None:
            return ""
        if isinstance(n, NavigableString):
            return norm(str(n))
        if isinstance(n, Tag):
            return norm(" ".join(n.stripped_strings))
        return ""

    # 1) Start scanning after the "상호작용 ( 총 ... 건 )" header, if present
    start = soup
    hdr = soup.find(string=re.compile(r"상호작용\s*\(\s*총\s*\d+\s*건\s*\)"))
    if hdr and getattr(hdr, "parent", None):
        start = hdr.parent

    # 2) Collect all '해당제품' text nodes in order
    markers = []
    for el in start.next_elements:
        if isinstance(el, NavigableString) and norm(str(el)) == "해당제품":
            markers.append(el)
        elif isinstance(el, Tag) and el.name in ("span", "a", "div", "p", "strong"):
            # Some parsers split text; guard by exact text of tag
            if node_text(el) == "해당제품":
                markers.append(el)

    if len(markers) < 2:
        return []

    # --- helpers -------------------------------------------------------------
    def nearest_left_ingredient(marker, stop=None) -> str:
        """
        Walk left (siblings first, then elements) until 'stop' (exclusive),
        prefer <a> text; otherwise take first non-empty text.
        """
        def valid_candidate(txt: str) -> bool:
            if not txt or txt in ("해당제품", "복사", "Image"):
                return False
            # Avoid headings/labels accidentally captured
            if "상호작용 ( 총" in txt or "성분 1 성분 2 내용" in txt:
                return False
            # Heuristic: ingredients are short-ish names without slashes/bullets
            if " / " in txt or "•" in txt:
                return False
            return True

        # 1) previous siblings within same parent (common case)
        sib = marker.previous_sibling
        hops = 0
        while sib is not None and sib is not stop and hops < 50:
            if isinstance(sib, Tag):
                # prefer last <a> inside the sibling
                anchors = sib.find_all("a")
                for cand in reversed(anchors):
                    t = norm(cand.get_text())
                    if valid_candidate(t):
                        return t
                t = node_text(sib)
                if valid_candidate(t):
                    return t
            elif isinstance(sib, NavigableString):
                t = norm(str(sib))
                if valid_candidate(t):
                    return t
            sib = sib.previous_sibling
            hops += 1

        # 2) previous_element walk, bounded by 'stop'
        e = marker.previous_element
        hops = 0
        while e is not None and e is not stop and hops < 200:
            if isinstance(e, Tag) and e.name == "a":
                t = norm(e.get_text())
                if valid_candidate(t):
                    return t
            elif isinstance(e, NavigableString):
                t = norm(str(e))
                if valid_candidate(t):
                    # ensure it's not crossing the boundary via containment
                    return t
            e = e.previous_element
            hops += 1
        return ""

    def collect_between(a, b) -> str:
        """
        Collect unique text from block items (<li>, <p>) strictly between a and b.
        Avoid double-counting by tracking element ids; no raw NavigableString grabs.
        """
        allowed = {"li", "p"}
        seen_ids = set()
        lines = []
        e = a
        steps = 0
        while e is not None and e is not b and steps < 1000:
            e = e.next_element
            if e is None or e is b:
                break
            if isinstance(e, Tag) and e.name in allowed and id(e) not in seen_ids:
                seen_ids.add(id(e))
                t = node_text(e)
                if t and t not in ("해당제품", "복사", "Image", "성분 1 성분 2 내용"):
                    lines.append(t)
            steps += 1
        # de-dup lines while preserving order
        uniq = []
        seen = set()
        for ln in lines:
            if ln not in seen:
                uniq.append(ln)
                seen.add(ln)
        return " ".join(uniq).strip()
    # ------------------------------------------------------------------------

    items = []
    # Pair strictly in twos: [0,1], [2,3], ...
    for i in range(0, len(markers) - 1, 2):
        mk1, mk2 = markers[i], markers[i + 1]

        # Ingredient 1: search left with stop=None or previous pair's mk2
        stop1 = markers[i - 1] if i - 1 >= 0 else None
        ing1 = nearest_left_ingredient(mk1, stop=stop1)

        # Ingredient 2: search left but STOP at mk1 so we don't cross into ing1
        ing2 = nearest_left_ingredient(mk2, stop=mk1)

        # Bound description by next pair's first marker
        next_first = markers[i + 2] if i + 2 < len(markers) else None
        content = collect_between(mk2, next_first)

        # Skip obvious bad rows
        if not ing1 or not ing2 or ing1 == ing2:
            continue

        items.append({"성분 1": ing1, "성분 2": ing2, "내용": content})

    return items
  
def parse_interaction_page(drug_cd: str) -> List[Dict[str, str]]:
    url = INTERACT_URL.format(drug_cd=drug_cd)
    soup = fetch(url)
    return parse_interaction_rows(soup)

def save_details_csv(rows: List[Dict[str, str]], path: str):
    if not rows:
        return
    fieldnames = ["drug_cd", "제품명", "성분", "효능·효과", "용법·용량", "source_url"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def save_interactions_csv(rows, path):
    """
    Always create the file (write header) even when rows == [].
    """
    fieldnames = ["drug_cd", "제품명", "성분 1", "성분 2", "내용"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
            
def crawl(drug_cds: List[str], delay: float = 1.0):
    all_details = []
    all_interactions = []
    for cd in drug_cds:
        print(f"[+] Crawling {cd}")
        detail = parse_detail_page(cd)
        all_details.append(detail)

        interactions = parse_interaction_page(cd)
        print(f"  parsed {len(interactions)} interaction rows for {cd}")
        # attach drug_cd + product name for context
        for row in interactions:
            row["drug_cd"] = cd
            row["제품명"] = detail.get("제품명", "")
        all_interactions.extend(interactions)
        time.sleep(delay)

    save_details_csv(all_details, f"{sys.argv[1]}_drug_details.csv")
    save_interactions_csv(all_interactions, f"{sys.argv[1]}_drug_interactions.csv")
    print(f"[✓] Saved: {sys.argv[1]}_drug_details.csv, {sys.argv[1]}_drug_interactions.csv")

if __name__ == "__main__":
    # Example: python crawl_healthkr.py 2021082400002 2018082200042 ...
    cds = sys.argv[1:] or ["2021082400002"]
    crawl(cds, delay=1.0)
