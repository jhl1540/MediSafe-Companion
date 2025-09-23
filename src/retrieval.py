
from __future__ import annotations
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .utils import _safe_json_parse

# Optional imports (all guarded)
LANGCHAIN_OPENAI_AVAILABLE = False
LC_PROMPTS_AVAILABLE = False
PUBCHEMPY_AVAILABLE = False
TAVILY_LC_AVAILABLE = False
TAVILY_NATIVE_AVAILABLE = False
REQUESTS_AVAILABLE = False
BS4_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI  # type: ignore
    LANGCHAIN_OPENAI_AVAILABLE = True
except Exception:
    pass

try:
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    LC_PROMPTS_AVAILABLE = True
except Exception:
    pass

try:
    import pubchempy as pcp  # type: ignore
    PUBCHEMPY_AVAILABLE = True
except Exception:
    pass

LC_TavilySearchResults = None
try:
    from langchain_tavily import TavilySearchResults as LC_TavilySearchResults  # type: ignore
    TAVILY_LC_AVAILABLE = True
except Exception:
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults as LC_TavilySearchResults  # type: ignore
        TAVILY_LC_AVAILABLE = True
    except Exception:
        pass

try:
    from tavily import TavilyClient  # type: ignore
    TAVILY_NATIVE_AVAILABLE = True
except Exception:
    pass

try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except Exception:
    pass

try:
    from bs4 import BeautifulSoup  # type: ignore
    BS4_AVAILABLE = True
except Exception:
    pass


# ------------------------- Core external helpers -------------------------

def llm_map_query_to_struct(query: str) -> Dict[str, Optional[str]]:
    """Use ChatOpenAI (if available) to map a drug name → identifiers.
    Returns: {name, synonyms[list], smiles, inchi, inchikey}
    """
    if not (LANGCHAIN_OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY")):
        return {}
    try:
        llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o"), temperature=0)
        prompt = (
            "You map user drug names (brand/generic) to chemical identifiers."
            "Return STRICT JSON with keys: name, synonyms (array), smiles, inchi, inchikey."
            "If unknown, return {}.Drug name: " + query
        )
        msg = llm.invoke(prompt)
        content = getattr(msg, "content", str(msg))
        data = _safe_json_parse(content)
        if not isinstance(data, dict):
            return {}
        syn = data.get("synonyms") or []
        if isinstance(syn, str):
            syn = [syn]
        return {
            "name": data.get("name"),
            "synonyms": syn,
            "smiles": data.get("smiles"),
            "inchi": data.get("inchi"),
            "inchikey": data.get("inchikey"),
        }
    except Exception:
        return {}


def pubchempy_resolve_name(query: str) -> Dict[str, Optional[str]]:
    if not PUBCHEMPY_AVAILABLE:
        return {}
    try:
        comps = pcp.get_compounds(query, "name")
        if not comps:
            return {}
        c = comps[0]
        smiles = getattr(c, "smiles", None) or getattr(c, "canonical_smiles", None)
        inchi = getattr(c, "inchi", None)
        inchikey = getattr(c, "inchikey", None)
        synonyms = []
        try:
            if getattr(c, "synonyms", None):
                synonyms = list(c.synonyms)
        except Exception:
            synonyms = []
        return {
            "name": getattr(c, "iupac_name", None) or query,
            "synonyms": synonyms or [query],
            "smiles": smiles,
            "inchi": inchi,
            "inchikey": inchikey,
        }
    except Exception:
        return {}


def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Run web search using **LangChain Tavily tool** when available.
    Falls back to native client, else returns [].
    Returns list of {title, url, snippet}.
    """
    if TAVILY_LC_AVAILABLE and os.environ.get("TAVILY_API_KEY") and LC_TavilySearchResults is not None:
        try:
            try:
                tool = LC_TavilySearchResults(k=max_results)  # type: ignore
            except TypeError:
                tool = LC_TavilySearchResults(max_results=max_results)  # type: ignore
            try:
                res = tool.invoke(query)  # type: ignore
            except Exception:
                res = tool.invoke({"query": query})  # type: ignore
            data = res
            if isinstance(res, str):
                data = _safe_json_parse(res)
            items: List[Dict[str, str]] = []
            iterable: List[Dict[str, Any]]
            if isinstance(data, dict) and "results" in data:
                iterable = data.get("results", [])
            elif isinstance(data, list):
                iterable = data
            else:
                iterable = []
            for r in iterable:
                title = r.get("title", "") if isinstance(r, dict) else ""
                url = r.get("url", "") if isinstance(r, dict) else ""
                snippet = r.get("content", "") if isinstance(r, dict) else ""
                items.append({"title": title, "url": url, "snippet": snippet})
            return items
        except Exception:
            pass

    if TAVILY_NATIVE_AVAILABLE and os.environ.get("TAVILY_API_KEY"):
        try:
            client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])  # type: ignore
            res = client.search(query=query, search_depth="advanced", max_results=max_results)  # type: ignore
            items = []
            for r in res.get("results", []):
                items.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                })
            return items
        except Exception:
            return []

    return []


# --- Health.kr helpers (KR brand page) ---

def find_healthkr_url_via_search(brand: str) -> Optional[str]:
    hits = tavily_search(f"site:health.kr {brand} 검색 결과 약품")
    for h in hits:
        url = h.get("url", "")
        if "health.kr" in url and "searchDrug/result_drug" in url:
            return url
    for h in hits:
        url = h.get("url", "")
        if "health.kr" in url:
            return url
    return None


def parse_healthkr_sections(html: str) -> Dict[str, Any]:
    data = {"brand": None, "ingredients_strength": None, "indications": None,
        "interactions": None, "manufacturer": None, "classification": None}
    if not (BS4_AVAILABLE and html):
        return data
    from bs4 import BeautifulSoup  # type: ignore
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("", strip=True)

    def grab(label: str, window: int = 200):
        i = text.find(label)
        if i == -1:
            return None
        return text[i : i + window]

    data["ingredients_strength"] = grab("성분/함량") or grab("성분 함량")
    data["indications"] = grab("효능/효과") or grab("효능") or grab("효능 효과")
    data["interactions"] = grab("상호작용")
    data["classification"] = grab("분류") or grab("효능군")
    title = soup.find("title")
    if title and title.text:
        data["brand"] = title.text.split("-")[0].strip()
    return data


def extract_with_llm_chatprompt(brand: str, url: str, html: str) -> Dict[str, Any]:
    if not (LANGCHAIN_OPENAI_AVAILABLE and LC_PROMPTS_AVAILABLE and os.environ.get("OPENAI_API_KEY")):
        return {}
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    from langchain_openai import ChatOpenAI  # type: ignore

    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o"), temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an information extraction agent for Korean drug monographs from health.kr. "
         "Return STRICT JSON with keys: brand, manufacturer, ingredients_strength, indications, interactions, url. "
         "Keep Korean field names as-is in content but JSON keys in English. Do not include markdown."),
        ("user",
         "Brand: {brand}\nURL: {url}\nHTML (possibly truncated):\n{html}\n\nExtract and return JSON only.")        
        
        
    ])
    snippet = html[:12000] if html else ""
    try:
        msgs = prompt.format_messages(brand=brand, url=url or "", html=snippet)
        resp = llm.invoke(msgs)
        content = getattr(resp, "content", str(resp))
        data = _safe_json_parse(content)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def write_brand_csv(path: str, row: Dict[str, Any]):
    import csv
    exists = os.path.exists(path)
    headers = ["timestamp", "brand", "url", "manufacturer", "ingredients_strength", "indications", "interactions"]
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not exists:
            w.writeheader()
        from datetime import datetime
        w.writerow({
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "brand": row.get("brand"),
            "url": row.get("url"),
            "manufacturer": row.get("manufacturer"),
            "ingredients_strength": row.get("ingredients_strength"),
            "indications": row.get("indications"),
            "interactions": row.get("interactions"),
        })


# --- STUB demo lookup retained for deterministic tests ---

def pubchem_resolve(query: str) -> Dict[str, Optional[str]]:
    demo = {
        "tylenol": {
            "name": "Acetaminophen",
            "synonyms": ["Tylenol", "Paracetamol"],
            "smiles": "CC(=O)NC1=CC=C(O)C=C1O",
            "inchikey": "RZVAJINKPMORJF-UHFFFAOYSA-N",
        },
        "acetaminophen": {
            "name": "Acetaminophen",
            "synonyms": ["Tylenol", "Paracetamol"],
            "smiles": "CC(=O)NC1=CC=C(O)C=C1O",
            "inchikey": "RZVAJINKPMORJF-UHFFFAOYSA-N",
        },
        "paracetamol": {
            "name": "Acetaminophen",
            "synonyms": ["Tylenol", "Paracetamol"],
            "smiles": "CC(=O)NC1=CC=C(O)C=C1O",
            "inchikey": "RZVAJINKPMORJF-UHFFFAOYSA-N",
        },
        "ethanol": {
            "name": "Ethanol",
            "synonyms": ["Alcohol"],
            "smiles": "CCO",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        },
    }
    key = query.strip().lower()
    return demo.get(key, {"name": query, "synonyms": [query], "smiles": None, "inchikey": None})

USERDB_HEADERS = [
    "timestamp","제품명1","성분1","성분2","성분3",
    "식약처분류","효능/효과1","효능/효과2","대상",
    "결과","제품명2","사유","출처"
]

def write_userdb_csv_row(path: str, row: Dict[str, Any]):
    import os, csv
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=USERDB_HEADERS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in USERDB_HEADERS})


# --- Optional external update stubs (DDInter/Cortellis) ---

def ddinter_fetch_updates() -> List[Dict[str, Any]]:
    return []


def cortellis_fetch_updates() -> List[Dict[str, Any]]:
    return []
