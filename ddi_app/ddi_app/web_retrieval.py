import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Optional, List, Dict, Tuple
from .db_csv import upsert_interaction
from .config import DB_CSV
import pandas as pd
import asyncio

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
DEFAULT_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "ko,en;q=0.9"}

class WebExtract(BaseModel):
    drug: str
    components: List[str] = []
    partner: Optional[str] = None
    interaction: Optional[str] = None
    severity: Optional[str] = None
    source: Optional[str] = None
    evidence: Optional[str] = None
    confidence: float = 0.7

class WebFetchError(Exception):
    pass

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.6, min=0.5, max=4),
       reraise=True, retry=retry_if_exception_type(WebFetchError))
async def _fetch_text(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        raise WebFetchError(str(e))

async def fetch_health_kr(drug: str) -> List[WebExtract]:
    base = "https://www.health.kr/searchDrug"
    search_url = f"{base}/search_result.asp?drug_name={drug}"
    out: List[WebExtract] = []
    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True) as ac:
        html = await _fetch_text(ac, search_url)
        soup = BeautifulSoup(html, "lxml")
        detail_candidates = []
        for a in soup.select('a[href*="result_drug.asp?drug_cd="]'):
            href = a.get("href")
            if href and href not in detail_candidates:
                detail_candidates.append(base + "/" + href.lstrip("/"))
        for url in detail_candidates[:2]:
            try:
                dhtml = await _fetch_text(ac, url)
            except Exception:
                continue
            dsoup = BeautifulSoup(dhtml, "lxml")
            name = dsoup.select_one(".tit_area h2, h1, .title")
            dname = name.get_text(strip=True) if name else drug
            comp_nodes = dsoup.select(".ingre, .comp, .table td, li")
            comps = []
            for n in comp_nodes:
                t = n.get_text(" ", strip=True)
                if any(k in t for k in ["mg", "정", "캡슐", "성분"]):
                    comps.append(t)
            comps = sorted(list({c for c in comps if len(c) < 120}))[:10]

            inter_url = None
            for a in dsoup.select('a[href*="result_interaction.asp?drug_cd="]'):
                inter_url = base + "/" + a.get("href").lstrip("/")
                break
            inter_desc = None
            if inter_url:
                try:
                    ihtml = await _fetch_text(ac, inter_url)
                    isoup = BeautifulSoup(ihtml, "lxml")
                    paras = [p.get_text(" ", strip=True) for p in isoup.select(".cnt, .contents, p, li")]
                    inter_desc = "; ".join(paras[:6]) if paras else None
                except Exception:
                    pass

            out.append(WebExtract(
                drug=dname or drug,
                components=comps,
                interaction=inter_desc,
                source="health.kr",
                evidence=(inter_url or url),
                confidence=0.75 if inter_desc else 0.65
            ))
    return out

async def fetch_ddinter(drug: str, partner: Optional[str]) -> List[WebExtract]:
    base = "https://ddinter.scbdd.com/ddinter"
    out: List[WebExtract] = []
    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, verify=False) as ac:
        try:
            if partner:
                url = f"{base}/drug-detail/DDInter14/"
                html = await _fetch_text(ac, url)
                soup = BeautifulSoup(html, "lxml")
                paras = [p.get_text(" ", strip=True) for p in soup.select("p, li")]
                desc = "; ".join(paras[:6]) if paras else None
                out.append(WebExtract(drug=drug, partner=partner, interaction=desc, source="ddinter", evidence=url, confidence=0.6))
            else:
                url = f"{base}/drug-list/"
                html = await _fetch_text(ac, url)
                soup = BeautifulSoup(html, "lxml")
                comps = [li.get_text(" ", strip=True) for li in soup.select("li")][:5]
                out.append(WebExtract(drug=drug, components=comps, source="ddinter", evidence=url, confidence=0.55))
        except Exception:
            return []
    return out

def score_confidence(source: str, base: float | None = None) -> float:
    if base is None:
        base = 0.6
    s = (source or "").lower()
    if s in {"db", "csv", "local"}:
        return max(base, 0.9)
    if s in {"health.kr", "drugbank", "hira", "fda", "korea mfds"}:
        return max(base, 0.8)
    if s in {"ddinter"}:
        return max(base, 0.65)
    if s in {"llm"}:
        return max(base, 0.5)
    return base

async def web_retrieve(drug: str, partner: Optional[str]) -> List[WebExtract]:
    tasks = [fetch_health_kr(drug)]
    if partner:
        tasks.append(fetch_ddinter(drug, partner))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    flat: List[WebExtract] = []
    for r in results:
        if isinstance(r, Exception):
            continue
        flat.extend(r)

    best: Dict[Tuple[str, Optional[str]], WebExtract] = {}
    for item in flat:
        key = (item.drug.lower(), item.partner.lower() if item.partner else None)
        if key not in best or (item.confidence or 0) > (best[key].confidence or 0):
            best[key] = item
    return list(best.values())