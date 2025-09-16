"""
LangGraph Sample (with Fallback): DDI (Drug–Drug Interaction) Chatbot Pipeline
-----------------------------------------------------------------------------
This file provides a runnable *skeleton* implementation of the node architecture
and workflow you described. It now **runs even if `langgraph` is not installed**
by using a small built‑in orchestrator with the same API surface we use here.

What you get:
  1) State definition for the graph
  2) Six nodes (N1–N6): Query Normalizer → Embed&Store → WebUpdater → DDI Ranker → Alternative Finder → Response Generator
  3) In-memory GraphDB fallback with a Neo4j-ready interface (commented examples)
  4) Stubs for external calls (PubChem, DDInter, Cortellis)
  5) Demo run + **light tests** (assertions) at the bottom

Dependencies (minimal):
  pip install numpy
Optional (auto-detected at runtime; safe to omit):
  pip install langgraph langchain-openai pubchempy tavily-python neo4j rdkit-pypi

Replace STUBs with your real integrations as you go.
"""
from __future__ import annotations

import os
import json
import math
import hashlib
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangGraph basics
from langgraph.graph import StateGraph, END

# Optional external deps (guarded)
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

try:
    from neo4j import GraphDatabase  # noqa: F401
    NEO4J_AVAILABLE = True
    
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    print(f'NEO4J_AVAILABLE: {NEO4J_AVAILABLE} ')
    print(f'NEO4J_URI: {NEO4J_URI} ')

except Exception:
    NEO4J_AVAILABLE = False

# ---------------------------------------------------------------------------
# 0) Orchestrator: use LangGraph if available, otherwise a tiny fallback
# ---------------------------------------------------------------------------
LANGGRAPH_AVAILABLE = False
END = "__END__"

try:
    from langgraph.graph import StateGraph as _LGStateGraph, END as _LG_END  # type: ignore
    LANGGRAPH_AVAILABLE = True
    END = _LG_END  # use real END sentinel
except Exception:
    LANGGRAPH_AVAILABLE = False


class _MiniStateGraph:
    """A minimal orchestrator compatible with the usage in this file.

    Methods:
      - add_node(name, fn)
      - set_entry_point(name)
      - add_edge(src, dst)
      - compile() → object with .invoke(state)
    """
    def __init__(self, _state_type):
        self._nodes: Dict[str, Any] = {}
        self._edges: Dict[str, List[str]] = {}
        self._entry: Optional[str] = None

    def add_node(self, name: str, fn):
        self._nodes[name] = fn

    def set_entry_point(self, name: str):
        self._entry = name

    def add_edge(self, src: str, dst: str):
        self._edges.setdefault(src, []).append(dst)

    def compile(self):
        graph = self

        class _Runner:
            def invoke(self, state: Dict[str, Any]):
                if graph._entry is None:
                    raise RuntimeError("No entry point set")
                cur = graph._entry
                s = dict(state)
                steps = 0
                while cur and cur != END:
                    if cur not in graph._nodes:
                        raise RuntimeError(f"Node '{cur}' not found")
                    fn = graph._nodes[cur]
                    res = fn(s)
                    if isinstance(res, dict):
                        s = res
                    nxts = graph._edges.get(cur, [])
                    if not nxts:
                        break
                    # Deterministic choice (first edge)
                    cur = nxts[0]
                    steps += 1
                    if steps > 10000:
                        raise RuntimeError("Graph appears to loop")
                return s

        return _Runner()


StateGraph = _LGStateGraph if LANGGRAPH_AVAILABLE else _MiniStateGraph


# ---------------------------------------------------------------------------
# Optional integrations (auto-detected)
# ---------------------------------------------------------------------------
LANGCHAIN_OPENAI_AVAILABLE = False
PUBCHEMPY_AVAILABLE = False
TAVILY_AVAILABLE = False

try:  # langchain-openai (ChatOpenAI)
    from langchain_openai import ChatOpenAI  # type: ignore
    LANGCHAIN_OPENAI_AVAILABLE = True
except Exception:
    LANGCHAIN_OPENAI_AVAILABLE = False

try:  # PubChemPy
    import pubchempy as pcp  # type: ignore
    PUBCHEMPY_AVAILABLE = True
except Exception:
    PUBCHEMPY_AVAILABLE = False

try:  # Tavily
    from tavily import TavilyClient  # type: ignore
    TAVILY_AVAILABLE = True
except Exception:
    TAVILY_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1) In-memory GraphDB fallback
# ---------------------------------------------------------------------------
class InMemoryGraphDB:
    """
    Simple directed multigraph-like store for:
      - nodes: key by inchikey or unique id
      - edges: interactions (drug1->drug2) with severity, mechanism, refs
      - query_records: list of query + resolution + web results (for audit)
    Used when no external DB is configured.
    """
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.interactions: List[Dict[str, Any]] = []
        self.query_records: List[Dict[str, Any]] = []

    def upsert_drug(self, drug: Dict[str, Any]):
        key = drug.get("inchikey") or drug.get("id") or drug.get("name")
        if not key:
            raise ValueError("Drug node missing identifier")
        existing = self.nodes.get(key, {})
        merged = {**existing, **drug}
        self.nodes[key] = merged
        return key

    def list_drugs(self) -> List[Dict[str, Any]]:
        return list(self.nodes.values())

    def add_interaction(self, d1: str, d2: str, severity: str, mechanism: str, refs: List[str]):
        self.interactions.append({
            "drug1": d1,
            "drug2": d2,
            "severity": severity,
            "mechanism": mechanism,
            "refs": refs,
        })

    def get_interactions_for(self, inchikey: str) -> List[Dict[str, Any]]:
        return [e for e in self.interactions if e["drug1"] == inchikey or e["drug2"] == inchikey]

    def add_query_record(self, record: Dict[str, Any]):
        """Persist query, normalization sources, and web results for auditing."""
        self.query_records.append(record)

    def ensure_demo_seed(self):
        """
        Seed with demo nodes/edges for Tylenol (Acetaminophen) and Ethanol
        including a severe interaction.
        """
        acet = {
            "name": "Acetaminophen",
            "synonyms": ["Tylenol", "Paracetamol"],
            "smiles": "CC(=O)NC1=CC=C(O)C=C1O",
            "inchikey": "RZVAJINKPMORJF-UHFFFAOYSA-N",
            "class": "Analgesic",
            "embedding": simple_embedding("Acetaminophen"),
        }
        ethanol = {
            "name": "Ethanol",
            "synonyms": ["Alcohol"],
            "smiles": "CCO",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "class": "CNS Depressant",
            "embedding": simple_embedding("Ethanol"),
        }
        salicylic = {
            "name": "Salicylic acid",
            "synonyms": ["2-Hydroxybenzoic acid"],
            "smiles": "C1=CC(=CC=C1C(=O)O)O",
            "inchikey": "YGSDEFSMJLZEOE-UHFFFAOYSA-N",
            "class": "NSAID",
            "embedding": simple_embedding("Salicylic acid"),
        }
        amyl_nitrite = {
            "name": "Amyl nitrite",
            "synonyms": ["Pentyl nitrite"],
            "smiles": "CCCCCONO",
            "inchikey": "CWHHHXQLPUJAES-UHFFFAOYSA-N",
            "class": "Vasodilator",
            "embedding": simple_embedding("Amyl nitrite"),
        }
        for d in (acet, ethanol, salicylic, amyl_nitrite):
            self.upsert_drug(d)

        # Add the severe interaction with a mechanism blurb and refs
        self.add_interaction(
            d1=acet["inchikey"],
            d2=ethanol["inchikey"],
            severity="High",
            mechanism=(
                "CYP2E1 induction by ethanol increases NAPQI formation from "
                "acetaminophen → hepatotoxicity"
            ),
            refs=["DDInter:DDInter14", "NIAAA guidance"],
        )


# ---------------------------------------------------------------------------
# 2) Utilities and STUBs
# ---------------------------------------------------------------------------
SEVERITY_ORDER = {"High": 3, "Moderate": 2, "Low": 1}


def simple_embedding(text: str, dim: int = 128) -> List[float]:
    """Deterministic toy embedding (hash → pseudo-random). Replace with your model."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand to dim via repeated hashing
    bytes_needed = dim
    buf = bytearray()
    seed = h
    while len(buf) < bytes_needed:
        seed = hashlib.sha256(seed).digest()
        buf.extend(seed)
    arr = np.frombuffer(bytes(buf[:dim]), dtype=np.uint8).astype(np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr.tolist()


def cosine_sim(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9
    return float(np.dot(va, vb) / denom)


# --- External helpers (optional) ---

def _safe_json_parse(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        # try to extract a JSON object if the model wrapped it in text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
        return {}


def llm_map_query_to_struct(query: str) -> Dict[str, Optional[str]]:
    """Use ChatOpenAI (if available) to map a drug name → identifiers.
    Returns keys: name, synonyms(list), smiles, inchi, inchikey.
    """
    if not (LANGCHAIN_OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY")):
        return {}
    try:
        llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o"), temperature=0)
        prompt = (
            "You map user drug names (brand/generic) to chemical identifiers.\n"
            "Return STRICT JSON with keys: name, synonyms (array), smiles, inchi, inchikey.\n"
            "If unknown, return {}.\n\nDrug name: "
            + query
        )
        msg = llm.invoke(prompt)  # LangChain returns an AIMessage with .content
        content = getattr(msg, "content", str(msg))
        data = _safe_json_parse(content)
        if not isinstance(data, dict):
            return {}
        # Normalize types
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
        smiles = getattr(c, "isomeric_smiles", None) or getattr(c, "canonical_smiles", None)
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
    if not (TAVILY_AVAILABLE and os.environ.get("TAVILY_API_KEY")):
        return []
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


# --- STUB demo lookup (kept for deterministic tests) ---

def pubchem_resolve(query: str) -> Dict[str, Optional[str]]:
    """
    STUB: Replace with real PubChem lookup if desired.
    Returns a dict with name, synonyms, smiles, inchikey if known.
    """
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


# ---------------------------------------------------------------------------
# 3) Graph State
# ---------------------------------------------------------------------------
class DDIState(TypedDict, total=False):
    user_query: str
    drug_query: str                     # normalized textual name
    normalized: Dict[str, Any]          # name/synonyms/smiles/inchikey
    smiles: Optional[str]
    inchikey: Optional[str]
    embedding: Optional[List[float]]
    refs: List[str]
    interactions: List[Dict[str, Any]]  # ranked list
    alternatives: List[Dict[str, Any]]  # list of {drug, alt, score}
    response: str


# ---------------------------------------------------------------------------
# 4) Shared resources (DB/session)
# ---------------------------------------------------------------------------
GRAPH = InMemoryGraphDB()
GRAPH.ensure_demo_seed()


# ---------------------------------------------------------------------------
# 5) Node Implementations
# ---------------------------------------------------------------------------
# Helper to merge resolution candidates with precedence

def _merge_resolution(query: str, *candidates: Dict[str, Any]) -> Dict[str, Any]:
    """Return the first candidate that has inchikey or smiles; fill missing fields from later ones."""
    base: Dict[str, Any] = {"name": query, "synonyms": [query], "smiles": None, "inchi": None, "inchikey": None}
    chosen = None
    for c in candidates:
        if not c:
            continue
        if (c.get("inchikey") or c.get("smiles")) and chosen is None:
            chosen = {**base, **c}
        else:
            # backfill any missing fields
            if chosen:
                for k, v in c.items():
                    if k not in chosen or chosen[k] in (None, [], ""):
                        chosen[k] = v
    return chosen or base


# N1: Query Normalizer (refactored to use LLM + PubChemPy + Web search, with fallbacks)

def n1_query_normalizer(state: DDIState) -> DDIState:
    query = state.get("user_query", "").strip()

    # Run optional resolvers
    llm_cand = llm_map_query_to_struct(query)
    pcp_cand = pubchempy_resolve_name(query)
    web_hits = tavily_search(query)

    # Keep stub for deterministic tests
    stub_cand = pubchem_resolve(query)

    # Choose best; precedence: PubChemPy → LLM → STUB (changeable by product policy)
    resolved = _merge_resolution(query, pcp_cand, llm_cand, stub_cand)

    # Persist audit trail in GraphDB
    GRAPH.add_query_record({
        "query": query,
        "resolved": resolved,
        "sources": {
            "llm": llm_cand,
            "pubchempy": pcp_cand,
            "stub": stub_cand,
        },
        "web_results": web_hits,
    })

    # Update state
    out = dict(state)
    out["drug_query"] = resolved.get("name") or query
    out["normalized"] = resolved
    out["smiles"] = resolved.get("smiles")
    out["inchikey"] = resolved.get("inchikey")
    return out


# N2: Embed & Store

def n2_embed_and_store(state: DDIState) -> DDIState:
    smiles = state.get("smiles")
    name = state.get("drug_query") or (state.get("normalized", {}).get("name"))
    inchikey = state.get("inchikey")

    # Make an embedding (deterministic text embedding here; swap for RDKit/mol model)
    emb = simple_embedding((name or smiles or "unknown"))

    # Upsert into GraphDB
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
    out["inchikey"] = key  # ensure we have a key
    return out


# N3: Web Updater (DDInter/Cortellis → Graph)

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


# N4: DDI Ranker

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
    # Aggregate unique refs for the response
    collated_refs: List[str] = []
    for r in ranked:
        for ref in r.get("refs", []):
            if ref not in collated_refs:
                collated_refs.append(ref)
    out["refs"] = collated_refs
    return out


# N5: Alternative Finder

def n5_alternative_finder(state: DDIState) -> DDIState:
    target_key = state.get("inchikey")
    target_node = GRAPH.nodes.get(target_key, {})
    target_emb = target_node.get("embedding")

    alternatives: List[Dict[str, Any]] = []

    # Exclude direct interactors from the alternative pool
    interactors = set()
    for e in GRAPH.get_interactions_for(target_key):
        interactors.add(e["drug1"]) ; interactors.add(e["drug2"])  # both ends

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


# N6: Response Generator

def n6_response_generator(state: DDIState) -> DDIState:
    name = state.get("drug_query", "the drug")
    inters = state.get("interactions", [])
    alts = state.get("alternatives", [])
    refs = state.get("refs", [])

    lines = []
    lines.append(f"DDI report for: {name}\n")

    if inters:
        lines.append("Possible interacting drugs (sorted by severity):")
        for r in inters:
            lines.append(
                f"  - {r['other_name']} (severity: {r['severity']})\n    mechanism: {r['mechanism']}"
            )
    else:
        lines.append("No interactions found in current knowledge base.")

    if alts:
        lines.append("\nAlternatives similar by embedding (top 5):")
        for a in alts:
            lines.append(f"  - {a['alt_name']} (similarity: {a['score']})")

    if refs:
        lines.append("\nReferences:")
        for r in refs:
            lines.append(f"  - {r}")

    lines.append(
        "\nDisclaimer: This assistant is for informational purposes only. "
        "Consult a licensed healthcare professional for medical advice."
    )

    out = dict(state)
    out["response"] = "\n".join(lines)
    return out


# ---------------------------------------------------------------------------
# 6) Build the (Mini)Graph
# ---------------------------------------------------------------------------
workflow = StateGraph(DDIState)
workflow.add_node("N1_QueryNormalizer", n1_query_normalizer)
workflow.add_node("N2_EmbedStore", n2_embed_and_store)
workflow.add_node("N3_WebUpdater", n3_web_updater)
workflow.add_node("N4_DDIRanker", n4_ddi_ranker)
workflow.add_node("N5_AltFinder", n5_alternative_finder)
workflow.add_node("N6_Response", n6_response_generator)

# Linear edges for the base path
workflow.set_entry_point("N1_QueryNormalizer")
workflow.add_edge("N1_QueryNormalizer", "N2_EmbedStore")
workflow.add_edge("N2_EmbedStore", "N3_WebUpdater")
workflow.add_edge("N3_WebUpdater", "N4_DDIRanker")
workflow.add_edge("N4_DDIRanker", "N5_AltFinder")
workflow.add_edge("N5_AltFinder", "N6_Response")
workflow.add_edge("N6_Response", END)

graph = workflow.compile()


# ---------------------------------------------------------------------------
# 7) Demo Runner + Light Tests
# ---------------------------------------------------------------------------

def ddinter_fetch_updates() -> List[Dict[str, Any]]:
    """STUB: return a list of new/updated interactions from DDInter."""
    return []

def cortellis_fetch_updates() -> List[Dict[str, Any]]:
    """STUB: return a list of new/updated interactions from Cortellis."""
    return []


def _run_smoke_tests():
    """Simple assertions to verify core behavior. These are not exhaustive."""
    # 1) Tylenol → should resolve to Acetaminophen and find Ethanol interaction (High)
    s1 = graph.invoke({"user_query": "Tylenol"})
    inters = s1.get("interactions", [])
    assert inters, "Expected at least one interaction for Tylenol demo seed"
    assert inters[0]["other_name"] == "Ethanol", "Ethanol should rank first by severity"
    assert inters[0]["severity"] == "High", "Expected 'High' severity for Tylenol↔Ethanol"

    # 2) Synonym mapping: Paracetamol → same inchikey
    s2 = graph.invoke({"user_query": "Paracetamol"})
    assert s2.get("inchikey") == "RZVAJINKPMORJF-UHFFFAOYSA-N", "Synonym should resolve to Acetaminophen"

    # 3) Unknown drug → no interactions, but alternatives list (from seed) should exist
    s3 = graph.invoke({"user_query": "UnknownDrugX"})
    assert s3.get("interactions") in ([], None), "Unknown drug shouldn't have seeded interactions"
    assert len(s3.get("alternatives", [])) >= 1, "Should propose some alternatives from seed pool"

    # 4) Query logs should be recorded in GraphDB (for audit)
    assert len(GRAPH.query_records) >= 3, "Expected query audit records to be stored"
    assert GRAPH.query_records[0].get("query") in {"Tylenol", "Paracetamol", "UnknownDrugX"}


if __name__ == "__main__":
    # Run smoke tests first
    _run_smoke_tests()

    # Demo inputs (kept from original; do not remove)
    demo_inputs = [
        {"user_query": "Tylenol"},
        {"user_query": "Ethanol"},
        {"user_query": "Medikinet"},
    ]

    for inp in demo_inputs:
        print("\n======================")
        print("INPUT:", inp)
        final_state = graph.invoke(inp)
        print(final_state.get("response", "<no response>"))
