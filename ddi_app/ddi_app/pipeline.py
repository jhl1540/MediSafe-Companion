# pipeline.py
import io
import asyncio
import pandas as pd
from typing import Optional, Dict, Any
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from .db_csv import read_db, write_db, get_monograph, upsert_interaction, search_db_for_drug, REQUIRED_CANONICAL
from .web_retrieval import web_retrieve, score_confidence
from .llm_backoff import llm_extract_ddi
from .neo4j_utils import Neo4jClient
from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from .ui import format_answer

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
        sub = df[\
            ((df["drug"].str.lower()==state.drug1.lower()) & (df["partner"].str.lower()==state.drug2.lower())) |\
            ((df["drug"].str.lower()==state.drug2.lower()) & (df["partner"].str.lower()==state.drug1.lower()))\
        ]
        if not sub.empty:
            row = sub.sort_values("updated_at", ascending=False).iloc[0].to_dict()
            ddi = {\
                "a": state.drug1,\
                "b": state.drug2,\
                "interaction": row.get("interaction"),\
                "severity": row.get("severity"),\
                "source": row.get("source"),\
                "confidence": float(row.get("confidence") or 0.9),\
                "evidence": row.get("evidence"),\
            }
    return state.copy(update={"monograph1": mono1, "monograph2": mono2, "ddi": ddi})

async def web_retrieve_node(state: AppState) -> AppState:
    import pandas as pd
    df = pd.read_json(io.StringIO(state.df_json)) if state.df_json else pd.DataFrame(columns=REQUIRED_CANONICAL)
    items = await web_retrieve(state.drug1, state.drug2)
    for ex in items:
        src = ex.source or "web"
        conf = score_confidence(src, ex.confidence)
        if ex.partner and ex.interaction:
            comp_list = ex.components or [""]
            for comp in comp_list:
                df = upsert_interaction(df, ex.drug or state.drug1, comp, ex.partner, ex.interaction or "", ex.severity or "", src, conf, ex.evidence or "")
        elif not state.drug2 and ex.components:
            for comp in ex.components:
                df = upsert_interaction(df, ex.drug or state.drug1, comp, "", "", "", src, conf, ex.evidence or "")
    write_db(df)
    mono1 = get_monograph(df, state.drug1)
    mono2 = get_monograph(df, state.drug2) if state.drug2 else None
    ddi = None
    if state.drug2:
        sub = df[\
            ((df["drug"].str.lower()==state.drug1.lower()) & (df["partner"].str.lower()==state.drug2.lower())) |\
            ((df["drug"].str.lower()==state.drug2.lower()) & (df["partner"].str.lower()==state.drug1.lower()))\
        ]
        if not sub.empty:
            row = sub.sort_values("confidence", ascending=False).iloc[0].to_dict()
            ddi = {\
                "a": state.drug1,\
                "b": state.drug2,\
                "interaction": row.get("interaction"),\
                "severity": row.get("severity"),\
                "source": row.get("source"),\
                "confidence": float(row.get("confidence") or 0.7),\
                "evidence": row.get("evidence"),\
            }
    return state.copy(update={"df_json": df.to_json(orient="records"), "monograph1": mono1, "monograph2": mono2, "ddi": ddi})

async def llm_backoff_node(state: AppState) -> AppState:
    import pandas as pd
    df = pd.read_json(io.StringIO(state.df_json)) if state.df_json else pd.DataFrame(columns=REQUIRED_CANONICAL)
    extracts = await llm_extract_ddi(state.drug1, state.drug2)
    for ex in extracts:
        comp_list = ex.components or [""]
        partner = ex.partner or (state.drug2 or "")
        if partner or ex.interaction:
            for comp in comp_list:
                df = upsert_interaction(df, ex.drug or state.drug1, comp, partner, ex.interaction or "", ex.severity or "", ex.source or "LLM", 0.5, ex.evidence or "")
    write_db(df)
    mono1 = get_monograph(df, state.drug1)
    mono2 = get_monograph(df, state.drug2) if state.drug2 else None
    ddi = None
    if state.drug2:
        sub = df[\
            ((df["drug"].str.lower()==state.drug1.lower()) & (df["partner"].str.lower()==state.drug2.lower())) |\
            ((df["drug"].str.lower()==state.drug2.lower()) & (df["partner"].str.lower()==state.drug1.lower()))\
        ]
        if not sub.empty:
            row = sub.sort_values("confidence", ascending=False).iloc[0].to_dict()
            ddi = {\
                "a": state.drug1,\
                "b": state.drug2,\
                "interaction": row.get("interaction"),\
                "severity": row.get("severity"),\
                "source": row.get("source"),\
                "confidence": float(row.get("confidence") or 0.5),\
                "evidence": row.get("evidence"),\
            }
    return state.copy(update={"df_json": df.to_json(orient="records"), "monograph1": mono1, "monograph2": mono2, "ddi": ddi})

async def write_neo4j_node(state: AppState) -> AppState:
    neo = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        neo.ensure_constraints()
        if state.monograph1 and state.monograph1.get("components"):
            neo.merge_components(state.drug1, state.monograph1["components"])
        if state.monograph2 and state.monograph2.get("components"):
            neo.merge_components(state.drug2, state.monograph2["components"])
        if state.drug2 and state.ddi and state.ddi.get("interaction"):
            neo.merge_ddi(state.ddi["a"], state.ddi["b"], state.ddi.get("interaction",""), state.ddi.get("severity",""),
                          state.ddi.get("source",""), float(state.ddi.get("confidence") or 0), state.ddi.get("evidence",""))
    finally:
        neo.close()
    return state

async def format_answer_node(state: AppState) -> AppState:
    md = format_answer(state.drug1, state.monograph1 or {}, state.drug2, state.monograph2, state.ddi)
    return state.copy(update={"answer_md": md})

def build_workflow() -> "StateGraph":
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
    workflow.add_conditional_edges("decide_source",
        lambda s: "from_db" if s.found_in_db else "need_web",
        {"from_db": "fetch_from_db", "need_web": "web_retrieve"})
    workflow.add_edge("web_retrieve", "llm_backoff")
    workflow.add_edge("fetch_from_db", "write_neo4j")
    workflow.add_edge("llm_backoff", "write_neo4j")
    workflow.add_edge("write_neo4j", "format_answer")
    workflow.add_edge("format_answer", END)
    return workflow