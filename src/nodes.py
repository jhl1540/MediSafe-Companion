from __future__ import annotations
import hashlib
import os
from typing import Any, Dict, List, Optional, TypedDict

from .state import DDIState, SEVERITY_ORDER
from .utils import simple_embedding, cosine_sim
from .graphdb import GRAPH
from .retrieval import (
    llm_map_query_to_struct,
    pubchempy_resolve_name,
    tavily_search,
    pubchem_resolve,
    find_healthkr_url_via_search,
    parse_healthkr_sections,
    extract_with_llm_chatprompt,
    write_brand_csv,
    ddinter_fetch_updates,
    cortellis_fetch_updates,
)


# ------------------------------ N0: KR Brand Fetcher ------------------------------

def _split_two_brands(q: str) -> tuple[str, Optional[str]]:
    for sep in [",", "/", "와", "및", "그리고", " and "]:
        if sep in q:
            a, b = [p.strip() for p in q.split(sep, 1)]
            return a, b or None
    toks = q.split()
    return (toks[0], " ".join(toks[1:])) if len(toks) >= 2 else (q, None)

def n0_kr_brand_fetcher(state: DDIState) -> DDIState:
    raw = (state.get("user_query") or "").strip()
    brand1, brand2 = _split_two_brands(raw)
    out = dict(state); out["brand1"]=brand1; out["brand2"]=brand2

    def _fetch_one(brand: str):
        url = find_healthkr_url_via_search(brand)
        html = None
        if url and os.environ.get("ENABLE_WEB","1")=="1":
            try:
                import requests
                r = requests.get(url, timeout=10)
                html = r.text if r.ok else None
            except Exception:
                pass
        parsed = extract_with_llm_chatprompt(brand, url or "", html or "") or parse_healthkr_sections(html or "")
        return url, {
            "brand": parsed.get("brand") or brand,
            "url": url,
            "manufacturer": parsed.get("manufacturer"),
            "ingredients_strength": parsed.get("ingredients_strength"),
            "indications": parsed.get("indications"),
            "interactions": parsed.get("interactions"),
            "classification": parsed.get("classification"),
        }

    url1, rec1 = _fetch_one(brand1)
    out["brand1_page_url"], out["brand1_scrape"] = url1, rec1
    if brand2:
        url2, rec2 = _fetch_one(brand2)
        out["brand2_page_url"], out["brand2_scrape"] = url2, rec2

    out["csv_userdb_path"]  = os.environ.get("KR_USERDB_CSV", "kr_user_db.csv")
    out["csv_brandlog_path"] = os.environ.get("HEALTHKR_CSV", "healthkr_drug_info.csv")
    return out

# ------------------------------ Helper to merge candidates ------------------------------

def _merge_resolution(query: str, *candidates: Dict[str, Any]) -> Dict[str, Any]:
    base: Dict[str, Any] = {"name": query, "synonyms": [query], "smiles": None, "inchi": None, "inchikey": None}
    chosen = None
    for c in candidates:
        if not c:
            continue
        if (c.get("inchikey") or c.get("smiles")) and chosen is None:
            chosen = {**base, **c}
        else:
            if chosen:
                for k, v in c.items():
                    if k not in chosen or chosen[k] in (None, [], ""):
                        chosen[k] = v
    return chosen or base


# ------------------------------ N1: Query Normalizer ------------------------------

def n1_query_normalizer(state: DDIState) -> DDIState:
    query = state.get("user_query", "").strip()

    llm_cand = llm_map_query_to_struct(query)
    pcp_cand = pubchempy_resolve_name(query)
    web_hits = tavily_search(query)
    stub_cand = pubchem_resolve(query)

    resolved = _merge_resolution(query, pcp_cand, llm_cand, stub_cand)

    GRAPH.add_query_record({
        "query": query,
        "resolved": resolved,
        "sources": {"llm": llm_cand, "pubchempy": pcp_cand, "stub": stub_cand},
        "web_results": web_hits,
    })

    out = dict(state)
    out["drug_query"] = resolved.get("name") or query
    out["normalized"] = resolved
    out["smiles"] = resolved.get("smiles")
    out["inchikey"] = resolved.get("inchikey")
    return out


# ------------------------------ N2: Embed & Store ------------------------------

def n2_embed_and_store(state: DDIState) -> DDIState:
    smiles = state.get("smiles")
    name = state.get("drug_query") or (state.get("normalized", {}).get("name"))
    inchikey = state.get("inchikey")

    emb = simple_embedding((name or smiles or "unknown"))

    node = {
        "name": name,
        "smiles": smiles,
        "inchikey": inchikey or f"TMP-{hashlib.md5((name or 'x').encode()).hexdigest()}",
        "embedding": emb,
        "class": None,
        "synonyms": state.get("normalized", {}).get("synonyms", []),
    }
    key = GRAPH.upsert_drug(node)

    out = dict(state)
    out["embedding"] = emb
    out["inchikey"] = key
    return out


# ------------------------------ N3: Web Updater ------------------------------

def n3_web_updater(state: DDIState) -> DDIState:
    updates: List[Dict[str, Any]] = []
    updates.extend(ddinter_fetch_updates())
    updates.extend(cortellis_fetch_updates())
    for u in updates:
        GRAPH.add_interaction(
            d1=u["drug1_inchikey"],
            d2=u["drug2_inchikey"],
            severity=u.get("severity", "Moderate"),
            mechanism=u.get("mechanism", "unknown"),
            refs=u.get("refs", []),
        )
    return state


# ------------------------------ N4: DDI Ranker ------------------------------

def n4_ddi_ranker(state: DDIState) -> DDIState:
    inchikey = state.get("inchikey")
    raw = GRAPH.get_interactions_for(inchikey)

    ranked: List[Dict[str, Any]] = []
    for e in raw:
        other = e["drug2"] if e["drug1"] == inchikey else e["drug1"]
        other_node = GRAPH.nodes.get(other)
        ranked.append({
            "other_inchikey": other,
            "other_name": other_node.get("name") if other_node else other,
            "severity": e["severity"],
            "mechanism": e["mechanism"],
            "refs": e.get("refs", []),
            "severity_rank": SEVERITY_ORDER.get(e["severity"], 0),
        })

    ranked.sort(key=lambda x: (-x["severity_rank"], x["other_name"]))

    out = dict(state)
    out["interactions"] = ranked
    collated_refs: List[str] = []
    for r in ranked:
        for ref in r.get("refs", []):
            if ref not in collated_refs:
                collated_refs.append(ref)
    out["refs"] = collated_refs
    return out


# ------------------------------ N5: Alternative Finder ------------------------------

def n5_alternative_finder(state: DDIState) -> DDIState:
    target_key = state.get("inchikey")
    target_node = GRAPH.nodes.get(target_key, {})
    target_emb = target_node.get("embedding")

    alternatives: List[Dict[str, Any]] = []
    interactors = set()
    for e in GRAPH.get_interactions_for(target_key):
        interactors.add(e["drug1"]); interactors.add(e["drug2"])  # both ends
    
    for key, node in GRAPH.nodes.items():
        if key == target_key or key in interactors:
            continue
        emb = node.get("embedding")
        score = cosine_sim(target_emb, emb) if (target_emb and emb) else 0.0
        alternatives.append({
            "source_inchikey": target_key,
            "source_name": target_node.get("name", target_key),
            "alt_inchikey": key,
            "alt_name": node.get("name", key),
            "score": round(float(score), 4),
        })

    alternatives.sort(key=lambda x: -x["score"])
    out = dict(state)
    out["alternatives"] = alternatives[:5]
    return out


# ------------------------------ N6: Response Generator ------------------------------

def n6_response_generator(state: DDIState) -> DDIState:
    name = state.get("drug_query", "the drug")
    inters = state.get("interactions", [])
    alts = state.get("alternatives", [])
    refs = state.get("refs", [])

    lines = []
    lines.append(f"DDI report for: {name}")

    if inters:
        lines.append("Possible interacting drugs (sorted by severity):")
        for r in inters:
            lines.append(
                f"  - {r['other_name']} (severity: {r['severity']}) mechanism: {r['mechanism']}"
            )
    else:
        lines.append("No interactions found in current knowledge base.")

    if alts:
        lines.append("Alternatives similar by embedding (top 5):")
        for a in alts:
            lines.append(f"  - {a['alt_name']} (similarity: {a['score']})")

    if refs:
        lines.append("References:")
        for r in refs:
            lines.append(f"  - {r}")

    lines.append(
        "Disclaimer: This assistant is for informational purposes only. "
        "Consult a licensed healthcare professional for medical advice."
    )

    out = dict(state)
    out["response"] = "".join(lines)
    return out
  
  
# ------------------------------ N7: CSV Writer ------------------------------  
  
import re
from .retrieval import write_userdb_csv_row

def _severity_symbol(sev: Optional[str]) -> str:
    return "❌" if sev in ("High","Contraindicated") else ("⚠️" if sev in ("Moderate","Low") else "")

def _find_node_by_name_or_syn(name: str) -> Optional[str]:
    name_l = name.strip().lower()
    for key, node in GRAPH.nodes.items():
        if str(node.get("name","")).lower() == name_l:
            return key
        for syn in node.get("synonyms", []) or []:
            if str(syn).lower() == name_l:
                return key
    return None





def n7_userdb_csv_writer(state: DDIState) -> DDIState:
    b1 = state.get("brand1") or state.get("drug_query") or ""
    b2 = state.get("brand2")
    s1 = state.get("brand1_scrape") or {}
    # split ingredients & indications
    def split3(s): 
        parts = re.split(r"[，,•·;\\n]+", s or "")
        parts = [p.strip() for p in parts if p.strip()]
        return (parts+[ "", "", "" ])[:3]
    def split2(s):
        parts = re.split(r"[。.\\.!?\\n,]+", s or "")
        parts = [p.strip() for p in parts if p.strip()]
        return (parts+[ "", "" ])[:2]
    ing1, ing2, ing3 = split3(s1.get("ingredients_strength"))
    eff1, eff2 = split2(s1.get("indications"))
    klass = (s1.get("classification") or "").strip()

    reason = ""; result_symbol = ""; refs = list(state.get("refs", []))
    if b2:
        k1 = state.get("inchikey")
        k2 = _find_node_by_name_or_syn(b2)
        if k1 and k2:
            for e in GRAPH.get_interactions_for(k1):
                other = e["drug2"] if e["drug1"] == k1 else e["drug1"]
                if other == k2:
                    reason = e.get("mechanism",""); result_symbol = _severity_symbol(e.get("severity"))
                    refs.extend(e.get("refs", [])); break
        if not reason and state.get("interactions"):
            top = state["interactions"][0]
            reason = top.get("mechanism",""); result_symbol = _severity_symbol(top.get("severity"))
            refs.extend(top.get("refs", []))

    sources = []
    if state.get("brand1_page_url"): sources.append(state["brand1_page_url"])
    if state.get("brand2_page_url"): sources.append(state["brand2_page_url"])
    for r in refs:
        if r and r not in sources: sources.append(str(r))

    row = {
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "제품명1": b1, "성분1": ing1, "성분2": ing2, "성분3": ing3,
        "식약처분류": klass, "효능/효과1": eff1, "효능/효과2": eff2,
        "대상": eff1, "결과": result_symbol, "제품명2": b2 or "", "사유": reason,
        "출처": "; ".join(sources),
    }
    try:
        write_userdb_csv_row(state.get("csv_userdb_path") or "kr_user_db.csv", row)
    except Exception:
        pass
    return state
  