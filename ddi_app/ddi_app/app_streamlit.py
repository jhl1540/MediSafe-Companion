"""
Project layout (single-file starter). You can split into modules later:
- app_streamlit.py               # Streamlit UI, runs the LangGraph pipeline

Before running:
1) pip install -U streamlit langgraph neo4j pandas python-dotenv httpx filelock openai pydantic beautifulsoup4 lxml tenacity
2) Set env vars:
   - OPENAI_API_KEY=...
   - NEO4J_URI=bolt://localhost:7687 (or neo4j+s://...)
   - NEO4J_USER=neo4j
   - NEO4J_PASSWORD=...
   - DB_CSV=./DB.csv   # path to your local CSV
3) Ensure Neo4j is up. The code will create basic constraints if missing.

CSV expectations (flexible, auto-mapped by case-insensitive heuristics):
Required columns (any of these aliases will be detected):
- drug (aliases: drug, drug_name, 제품명, 약품명)
- component (aliases: component, 성분, 성분1, 성분_리스트)
- partner (aliases: partner, 상대약물, 상호작용상대, 제품명2)
- interaction (aliases: interaction, 상호작용, 설명, 사유)
- severity (aliases: severity, 등급, 결과)
- source (aliases: source, 출처)
- updated_at (aliases: updated_at, 업데이트, 수정일)
- confidence (auto-added)
- evidence (auto-added)

Now includes:
- Async web retrieval (httpx) from known sources (health.kr placeholder, ddinter placeholder) → HTML parse (bs4) → structure
- LLM backoff (json-extraction) only if web fails
- Neo4j schema extended: (Drug)-[:HAS_COMPONENT]->(Component), (Drug)-[:INTERACTS_WITH {severity, evidence, source, confidence}]->(Drug)
- Source labeling and confidence scoring (DB>Web>LLM)
"""

import os
import io
import json
import re
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import streamlit as st
from filelock import FileLock
from pydantic import BaseModel, Field
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from neo4j import GraphDatabase

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# OpenAI
from openai import OpenAI

# ---------------------------
# Environment & Clients
# ---------------------------
load_dotenv()
DB_CSV = os.getenv("DB_CSV", "data/DB.csv")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Streamlit UI setup
# ---------------------------
st.set_page_config(page_title="약물 상호작용 분석기", layout="wide")
st.title("💊 약물 상호작용 분석기")

st.markdown(
    """
#### 💬 어떤 약물(약품)에 대해 궁금하세요? 또는 두 약물의 상호관계를 알고 싶으신가요?
- **한 가지 약물**만 궁금하시면 👉 왼쪽 입력칸에만 입력해 주세요.  
- **약물 간 상호작용**이 궁금하면 👉 오른쪽 입력칸도 함께 입력해 주세요.
"""
)

col1, col2 = st.columns(2)
with col1:
    drug1 = st.text_input("🩺 약물(약품) 1", placeholder="예: 타이레놀").strip()
with col2:
    drug2 = st.text_input("🩺 약물(약품) 2", placeholder="예: 이부프로펜").strip()

btn = st.button("🔍 분석하기")

# ---------------------------
# CSV DB utilities
# ---------------------------
COLUMN_ALIASES = {
    "drug": ["drug", "drug_name", "제품명", "약품명"],
    "component": ["component", "components", "성분", "성분1", "성분_리스트"],
    "partner": ["partner", "상대약물", "상호작용상대", "제품명2", "상대"],
    "interaction": ["interaction", "상호작용", "설명", "사유"],
    "severity": ["severity", "등급", "결과"],
    "source": ["source", "출처"],
    "updated_at": ["updated_at", "업데이트", "수정일"],
    "confidence": ["confidence"],
    "evidence": ["evidence", "근거"],
}

REQUIRED_CANONICAL = [
    "drug",
    "component",
    "partner",
    "interaction",
    "severity",
    "source",
    "updated_at",
    "confidence",
    "evidence",
]


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a.lower() in lower_cols:
                mapping[lower_cols[a.lower()]] = canon
                break
    df = df.rename(columns=mapping)
    for c in REQUIRED_CANONICAL:
        if c not in df.columns:
            df[c] = None
    return df[REQUIRED_CANONICAL]


def read_db() -> pd.DataFrame:
    if not os.path.exists(DB_CSV):
        return pd.DataFrame(columns=REQUIRED_CANONICAL)
    with FileLock(DB_CSV + ".lock"):
        df = pd.read_csv(DB_CSV)
    df = _canonicalize_columns(df)
    return df


def write_db(df: pd.DataFrame) -> None:
    df = _canonicalize_columns(df)
    with FileLock(DB_CSV + ".lock"):
        df.to_csv(DB_CSV, index=False)


def search_db_for_drug(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not name:
        return df.iloc[0:0]
    pat = re.escape(name.lower())
    mask = df["drug"].fillna("").str.lower().str.contains(pat)
    mask_partner = df["partner"].fillna("").str.lower().str.contains(pat)
    return df[mask | mask_partner]


def get_monograph(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    subset = search_db_for_drug(df, name)
    comps = sorted({c for c in subset["component"].dropna().astype(str).tolist() if c})
    interactions = subset.to_dict(orient="records")
    return {"drug": name, "components": comps, "interactions": interactions}


def upsert_interaction(
    df: pd.DataFrame,
    drug: str,
    component: str,
    partner: str,
    interaction: str,
    severity: str = "",
    source: str = "",
    confidence: float = 0.5,
    evidence: str = "",
) -> pd.DataFrame:
    now = datetime.utcnow().isoformat()
    row = {
        "drug": drug,
        "component": component,
        "partner": partner,
        "interaction": interaction,
        "severity": severity,
        "source": source,
        "updated_at": now,
        "confidence": confidence,
        "evidence": evidence,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df

# ---------------------------
# Web retrieval (httpx + bs4) — async & robust
# ---------------------------
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
    confidence: float = 0.7  # default for web


class WebFetchError(Exception):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.6, min=0.5, max=4), reraise=True, retry=retry_if_exception_type(WebFetchError))
async def _fetch_text(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        raise WebFetchError(str(e))


async def fetch_health_kr(drug: str) -> List[WebExtract]:
    """Placeholder parser for health.kr (의약품안전나라와 유사 구성). Structure may change.
    We try to be resilient: search page then detail (if needed).
    """
    base = "https://www.health.kr/searchDrug"
    search_url = f"{base}/search_result.asp?drug_name={httpx.QueryParams({'drug_name': drug}).get('drug_name')}"
    detail_candidates: List[str] = []
    out: List[WebExtract] = []

    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True) as ac:
        html = await _fetch_text(ac, search_url)
        soup = BeautifulSoup(html, "lxml")
        # Heuristic: find detail links
        for a in soup.select('a[href*="result_drug.asp?drug_cd="]'):
            href = a.get('href')
            if href and href not in detail_candidates:
                detail_candidates.append(base + "/" + href.lstrip("/"))
        # Visit top 1-2 candidates
        for url in detail_candidates[:2]:
            try:
                dhtml = await _fetch_text(ac, url)
            except Exception:
                continue
            dsoup = BeautifulSoup(dhtml, "lxml")
            # Very rough extraction heuristics
            name = dsoup.select_one(".tit_area h2, h1, .title")
            dname = name.get_text(strip=True) if name else drug
            comp_nodes = dsoup.select(".ingre, .comp, .table td, li")
            comps = []
            for n in comp_nodes:
                t = n.get_text(" ", strip=True)
                if any(k in t for k in ["mg", "정", "캡슐", "성분"]):
                    comps.append(t)
            comps = sorted(list({c for c in comps if len(c) < 120}))[:10]

            # Interaction page (if exists)
            inter_url = None
            for a in dsoup.select('a[href*="result_interaction.asp?drug_cd="]'):
                inter_url = base + "/" + a.get('href').lstrip('/')
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

            out.append(
                WebExtract(
                    drug=dname or drug,
                    components=comps,
                    interaction=inter_desc,
                    source="health.kr",
                    evidence=(inter_url or url),
                    confidence=0.75 if inter_desc else 0.65,
                )
            )
    return out


async def fetch_ddinter(drug: str, partner: Optional[str]) -> List[WebExtract]:
    """Placeholder parser for ddinter.scbdd.com (구조 자주 변경됨). We try simple scrape.
    If partner is given, attempt to read pair-wise page.
    """
    base = "https://ddinter.scbdd.com/ddinter"
    out: List[WebExtract] = []
    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, verify=False) as ac:
        # Note: verify=False due to past cert issues; consider fixing cert chain in production.
        try:
            if partner:
                # This site often uses drug-detail pages; we just record a generic summary.
                url = f"{base}/drug-detail/DDInter14/"  # placeholder entry point
                html = await _fetch_text(ac, url)
                soup = BeautifulSoup(html, "lxml")
                # fake heuristic summary
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


async def web_retrieve(drug: str, partner: Optional[str]) -> List[WebExtract]:
    tasks = [fetch_health_kr(drug)]
    if partner:
        tasks.append(fetch_ddinter(drug, partner))
    results = await asyncio.gather(*[t for t in tasks], return_exceptions=True)
    flat: List[WebExtract] = []
    for r in results:
        if isinstance(r, Exception):
            continue
        flat.extend(r)
    # Coalesce per (drug, partner) with best confidence
    best: Dict[Tuple[str, Optional[str]], WebExtract] = {}
    for item in flat:
        key = (item.drug.lower(), item.partner.lower() if item.partner else None)
        if key not in best or (item.confidence or 0) > (best[key].confidence or 0):
            best[key] = item
    return list(best.values())

# ---------------------------
# LLM schema & backoff extraction
# ---------------------------
class DDIExtract(BaseModel):
    drug: str = Field(..., description="Drug name")
    components: List[str] = Field(default_factory=list)
    partner: Optional[str] = Field(None, description="Other drug if a pair query")
    interaction: Optional[str] = None
    severity: Optional[str] = None
    source: Optional[str] = None
    evidence: Optional[str] = None
    confidence: Optional[float] = 0.5


async def llm_extract_ddi(drug: str, partner: Optional[str]) -> List[DDIExtract]:
    system = "You are a clinical pharmacist. Return concise, factual JSON. Include `source` and short `evidence` text when you can."
    user = (
        f"Extract components and, if partner is given, the DDI between {drug}"
        + (f" and {partner}." if partner else ".")
        + " Return JSON array of objects with keys: drug, components, partner, interaction, severity, source, evidence, confidence."
    )

    def _call():
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    content = await asyncio.to_thread(_call)
    try:
        data = json.loads(content)
        arr = data if isinstance(data, list) else data.get("items") or data.get("data") or []
    except Exception:
        arr = []
    out: List[DDIExtract] = []
    for item in arr:
        try:
            out.append(DDIExtract(**item))
        except Exception:
            continue
    if not out:
        out.append(DDIExtract(drug=drug, partner=partner, components=[], interaction=None))
    return out

# ---------------------------
# Confidence utils
# ---------------------------
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

# ---------------------------
# LangGraph state & nodes
# ---------------------------
class AppState(BaseModel):
    drug1: str
    drug2: Optional[str] = None
    df_json: Optional[str] = None
    found_in_db: bool = False
    monograph1: Optional[Dict[str, Any]] = None
    monograph2: Optional[Dict[str, Any]] = None
    ddi: Optional[Dict[str, Any]] = None
    answer_md: Optional[str] = None


async def load_db_node(state: AppState) -> AppState:
    df = read_db()
    return state.copy(update={"df_json": df.to_json(orient="records")})


async def decide_source_node(state: AppState) -> AppState:
    df = pd.read_json(io.StringIO(state.df_json)) if state.df_json else pd.DataFrame(columns=REQUIRED_CANONICAL)
    has1 = not search_db_for_drug(df, state.drug1).empty
    has2 = (not search_db_for_drug(df, state.drug2).empty) if state.drug2 else True
    return state.copy(update={"found_in_db": bool(has1 and has2)})


async def fetch_from_db_node(state: AppState) -> AppState:
    df = pd.read_json(io.StringIO(state.df_json)) if state.df_json else pd.DataFrame(columns=REQUIRED_CANONICAL)
    mono1 = get_monograph(df, state.drug1)
    mono2 = get_monograph(df, state.drug2) if state.drug2 else None
    ddi = None
    if state.drug2:
        sub = df[
            ((df["drug"].str.lower()==state.drug1.lower()) & (df["partner"].str.lower()==state.drug2.lower())) |
            ((df["drug"].str.lower()==state.drug2.lower()) & (df["partner"].str.lower()==state.drug1.lower()))
        ]
        if not sub.empty:
            row = sub.sort_values("updated_at", ascending=False).iloc[0].to_dict()
            ddi = {
                "a": state.drug1,
                "b": state.drug2,
                "interaction": row.get("interaction"),
                "severity": row.get("severity"),
                "source": row.get("source"),
                "confidence": float(row.get("confidence") or 0.9),
                "evidence": row.get("evidence"),
            }
    return state.copy(update={"monograph1": mono1, "monograph2": mono2, "ddi": ddi})


async def web_retrieve_node(state: AppState) -> AppState:
    df = pd.read_json(io.StringIO(state.df_json)) if state.df_json else pd.DataFrame(columns=REQUIRED_CANONICAL)
    items = await web_retrieve(state.drug1, state.drug2)
    # Upsert best available evidence
    for ex in items:
        src = ex.source or "web"
        conf = score_confidence(src, ex.confidence)
        if ex.partner and ex.interaction:
            comp_list = ex.components or [""]
            for comp in comp_list:
                df = upsert_interaction(
                    df,
                    drug=ex.drug or state.drug1,
                    component=comp,
                    partner=ex.partner,
                    interaction=ex.interaction or "",
                    severity=ex.severity or "",
                    source=src,
                    confidence=conf,
                    evidence=ex.evidence or "",
                )
        elif not state.drug2 and ex.components:
            # monograph-only; store components (no partner)
            for comp in ex.components:
                df = upsert_interaction(
                    df,
                    drug=ex.drug or state.drug1,
                    component=comp,
                    partner="",
                    interaction="",
                    severity="",
                    source=src,
                    confidence=conf,
                    evidence=ex.evidence or "",
                )
    write_db(df)

    # Build monographs & pair
    mono1 = get_monograph(df, state.drug1)
    mono2 = get_monograph(df, state.drug2) if state.drug2 else None
    ddi = None
    if state.drug2:
        sub = df[
            ((df["drug"].str.lower()==state.drug1.lower()) & (df["partner"].str.lower()==state.drug2.lower())) |
            ((df["drug"].str.lower()==state.drug2.lower()) & (df["partner"].str.lower()==state.drug1.lower()))
        ]
        if not sub.empty:
            row = sub.sort_values("confidence", ascending=False).iloc[0].to_dict()
            ddi = {
                "a": state.drug1,
                "b": state.drug2,
                "interaction": row.get("interaction"),
                "severity": row.get("severity"),
                "source": row.get("source"),
                "confidence": float(row.get("confidence") or 0.7),
                "evidence": row.get("evidence"),
            }
    return state.copy(update={"df_json": df.to_json(orient="records"), "monograph1": mono1, "monograph2": mono2, "ddi": ddi})


async def llm_backoff_node(state: AppState) -> AppState:
    # If web retrieval failed to produce useful info, use LLM to extract and store
    df = pd.read_json(io.StringIO(state.df_json)) if state.df_json else pd.DataFrame(columns=REQUIRED_CANONICAL)
    extracts = await llm_extract_ddi(state.drug1, state.drug2)
    for ex in extracts:
        comp_list = ex.components or [""]
        partner = ex.partner or (state.drug2 or "")
        if partner or ex.interaction:
            for comp in comp_list:
                df = upsert_interaction(
                    df,
                    drug=ex.drug or state.drug1,
                    component=comp,
                    partner=partner,
                    interaction=ex.interaction or "",
                    severity=ex.severity or "",
                    source=ex.source or "LLM",
                    confidence=score_confidence("LLM", ex.confidence),
                    evidence=ex.evidence or "",
                )
    write_db(df)

    mono1 = get_monograph(df, state.drug1)
    mono2 = get_monograph(df, state.drug2) if state.drug2 else None
    ddi = None
    if state.drug2:
        sub = df[
            ((df["drug"].str.lower()==state.drug1.lower()) & (df["partner"].str.lower()==state.drug2.lower())) |
            ((df["drug"].str.lower()==state.drug2.lower()) & (df["partner"].str.lower()==state.drug1.lower()))
        ]
        if not sub.empty:
            row = sub.sort_values("confidence", ascending=False).iloc[0].to_dict()
            ddi = {
                "a": state.drug1,
                "b": state.drug2,
                "interaction": row.get("interaction"),
                "severity": row.get("severity"),
                "source": row.get("source"),
                "confidence": float(row.get("confidence") or 0.5),
                "evidence": row.get("evidence"),
            }
    return state.copy(update={"df_json": df.to_json(orient="records"), "monograph1": mono1, "monograph2": mono2, "ddi": ddi})


# ---------------------------
# Neo4j utilities (extended schema)
# ---------------------------
class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def ensure_constraints(self):
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Component) REQUIRE c.name IS UNIQUE")

    def merge_components(self, drug: str, components: List[str]):
      if not components:
          return
      q = (
          "MERGE (d:Drug {name:$drug})\n"
          "WITH d\n"
          "UNWIND $components AS cname\n"
          "MERGE (c:Component {name:cname})\n"
          "MERGE (d)-[:HAS_COMPONENT]->(c)"
      )
      with self.driver.session() as s:
          s.run(q, drug=drug, components=components)

    def merge_ddi(self, a: str, b: str, interaction: str, severity: str, source: str, confidence: float, evidence: str):
      q = (
          "MERGE (a:Drug {name:$a})\n"
          "MERGE (b:Drug {name:$b})\n"
          "MERGE (a)-[r:INTERACTS_WITH]->(b)\n"
          "ON CREATE SET r.first_seen = timestamp()\n"
          "SET r.last_seen = timestamp(), r.description=$desc, r.severity=$sev, "
          "r.source=$src, r.confidence=$conf, r.evidence=$evid"
      )
      q2 = q.replace("(a)-[r:INTERACTS_WITH]->(b)", "(b)-[r:INTERACTS_WITH]->(a)")
      with self.driver.session() as s:
          s.run(q, a=a, b=b, desc=interaction, sev=severity, src=source, conf=float(confidence or 0), evid=evidence)
          s.run(q2, a=a, b=b, desc=interaction, sev=severity, src=source, conf=float(confidence or 0), evid=evidence)


async def write_neo4j_node(state: AppState) -> AppState:
    neo = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        neo.ensure_constraints()
        # Components
        if state.monograph1 and state.monograph1.get("components"):
            neo.merge_components(state.drug1, state.monograph1["components"])
        if state.monograph2 and state.monograph2.get("components"):
            neo.merge_components(state.drug2, state.monograph2["components"])
        # DDI
        if state.drug2 and state.ddi and state.ddi.get("interaction"):
            neo.merge_ddi(
                state.ddi["a"],
                state.ddi["b"],
                interaction=state.ddi.get("interaction", ""),
                severity=state.ddi.get("severity", ""),
                source=state.ddi.get("source", ""),
                confidence=float(state.ddi.get("confidence") or 0),
                evidence=state.ddi.get("evidence", ""),
            )
    finally:
        neo.close()
    return state


async def format_answer_node(state: AppState) -> AppState:
    mono1 = state.monograph1 or {}
    mono2 = state.monograph2 or None
    ddi = state.ddi

    def monograph_md(m):
      comps = ", ".join(m.get('components', []) or []) or "정보 없음"
      return (
          f"**구성 성분:** {comps}\n\n"
          f"**DB 내 상호작용 레코드 수:** {len(m.get('interactions', []))}\n"
      )

    sections = []
    sections.append(f"### 📌 약물 1: {state.drug1}" + monograph_md(mono1))
    if mono2:
        sections.append(f"### 📌 약물 2: {state.drug2}" + monograph_md(mono2))

    if state.drug2:
        sections.append("### 💥 두 약물의 상호작용")
        if ddi and ddi.get("interaction"):
            src = ddi.get('source','DB/LLM/WEB')
            conf = ddi.get('confidence')
            ev = ddi.get('evidence')
            sections.append(
                (
                    f"- ✅ **함께 복용 가능 여부/주의:** {ddi.get('severity','미상')}"
                    f"- ❗ **요약:** {ddi.get('interaction')}"
                    f"- 📚 **출처:** {src}  |  **신뢰도:** {conf if conf is not None else 'N/A'}"
                    + (f"- 🔎 **근거:** {ev}" if ev else "")
                )
            )
        else:
            sections.append("- DB/웹에서 명확한 상호작용 정보를 찾지 못했습니다. 최신 자료를 기반으로 추가 확인이 필요합니다.")
    md = "\n\n".join(sections)
    return state.copy(update={"answer_md": md})


# ---------------------------
# Build the LangGraph (with web + LLM backoff)
# ---------------------------
workflow = StateGraph(AppState)
workflow.add_node("load_db", load_db_node)
workflow.add_node("decide_source", decide_source_node)
workflow.add_node("fetch_from_db", fetch_from_db_node)
workflow.add_node("web_retrieve", web_retrieve_node)
workflow.add_node("llm_backoff", llm_backoff_node)
workflow.add_node("write_neo4j", write_neo4j_node)
workflow.add_node("format_answer", format_answer_node)

workflow.set_entry_point("load_db")
workflow.add_edge("load_db", "decide_source")
workflow.add_conditional_edges(
    "decide_source",
    lambda s: "from_db" if s.found_in_db else "need_web",
    {"from_db": "fetch_from_db", "need_web": "web_retrieve"},
)
# If web didn't populate useful info, we still proceed (LLM backoff)
workflow.add_edge("web_retrieve", "llm_backoff")
# Merge path to Neo4j write
workflow.add_edge("fetch_from_db", "write_neo4j")
workflow.add_edge("llm_backoff", "write_neo4j")
workflow.add_edge("write_neo4j", "format_answer")
workflow.add_edge("format_answer", END)


# Initialize Async checkpointer + compile the app within the event loop once
async def _ensure_app_compiled():
    if "app_compiled" in st.session_state:
        return
    cm = AsyncSqliteSaver.from_conn_string("langgraph_checkpoints.db")
    saver = await cm.__aenter__()  # enter async context manager
    st.session_state["_cp_cm"] = cm      # 보관(필요 시 종료)
    st.session_state["checkpointer"] = saver
    st.session_state["app_compiled"] = workflow.compile(checkpointer=saver)

# 모듈 로드 시 1회 보장(스트림릿 리런 대응)
try:
    asyncio.get_running_loop()
    asyncio.run(_ensure_app_compiled())
except RuntimeError:
    asyncio.run(_ensure_app_compiled())


# Ensure a stable thread_id for LangGraph checkpoints per Streamlit session
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"ui-{uuid4()}"
# Optional: namespace to separate flows (useful when you add more graphs)
st.session_state.setdefault("checkpoint_ns", "streamlit-ddi")

# ---------------------------
# Run on click
# ---------------------------
if btn:
    if not drug1:
        st.warning("⚠️ 약물 1은 반드시 입력해야 합니다.")
        st.stop()

    with st.spinner("💬 분석 파이프라인 실행 중..."):
        init = AppState(drug1=drug1, drug2=drug2 or None)
        try:
          asyncio.get_running_loop()
          asyncio.run(_ensure_app_compiled())
        except RuntimeError:
          asyncio.run(_ensure_app_compiled())

        app = st.session_state["app_compiled"]
        
        final_state = asyncio.run(
            app.ainvoke(
                init,
                config={
                    "configurable": {
                        "thread_id": st.session_state["thread_id"],
                        "checkpoint_ns": st.session_state["checkpoint_ns"],
                        # you may also pass a custom checkpoint_id if you want manual versioning
                    }
                },
            )
        )

    st.markdown("---")
    if drug2:
        try:
            parts = final_state.answer_md.split(f"### 📌 약물 2: {drug2}")
            left = parts[0]
            rest = f"### 📌 약물 2: {drug2}" + (parts[1] if len(parts) > 1 else "")
            right, *tail = rest.split("### 💥 두 약물의 상호작용")
            with col1:
                st.markdown(left, unsafe_allow_html=True)
            with col2:
                st.markdown(right, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### 💥 두 약물의 상호작용\n" + (tail[0] if tail else ""), unsafe_allow_html=True)
        except Exception:
            st.markdown(final_state.answer_md, unsafe_allow_html=True)
    else:
        with col1:
            st.markdown(final_state.answer_md, unsafe_allow_html=True)

    with st.expander("🔎 처리 상세 보기 (JSON)"):
        st.json({
            "found_in_db": final_state.found_in_db,
            "monograph1": final_state.monograph1,
            "monograph2": final_state.monograph2,
            "ddi": final_state.ddi,
        })

else:
    st.info("ℹ️ 위에 약물명을 입력하고 '🔍 분석하기' 버튼을 눌러주세요.")
