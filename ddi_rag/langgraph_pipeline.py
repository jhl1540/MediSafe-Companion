# langgraph_pipeline.py
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import asyncio

from parsers import search_healthkr_drug_cd, fetch_healthkr_fields, parse_ddi2_pair
from graph_store import GraphStore, NeoCfg

import os
from db_io import load_db, save_db, upsert_single, upsert_pair, has_single, has_pair
USE_LOCAL_DB = os.getenv("USE_LOCAL_DB", "true").lower() in ("1","true","yes")
USE_LLM_SEARCH = os.getenv("USE_LLM_SEARCH", "true").lower() in ("1","true","yes")


from llm_nodes import llm_plan
from llm_websearch import llm_search_then_summarize
import settings


# ---------------- State ----------------
class DDIState(BaseModel):
    messages: list[dict[str,str]] = Field(default_factory=list)
    query: str = ""
    drugA: str = ""
    drugB: str = ""
    healthA: dict = Field(default_factory=dict)
    healthB: dict = Field(default_factory=dict)
    ddi: dict = Field(default_factory=dict)
    plan: dict = Field(default_factory=dict)
    web_search_notes: str = ""          # <-- keep LLM notes here
    answer: str = ""


async def llm_plan_node(state: DDIState) -> DDIState:
    text = state.query if state.query else state.messages[-1]["content"]
    plan = await _to_thread(llm_plan, text)  # off the event loop
    state.plan = plan
    drugs = plan.get("drugs", [])
    state.drugA = drugs[0] if len(drugs) >= 1 else ""
    state.drugB = drugs[1] if len(drugs) >= 2 else ""
    return state

def selective_retrieval_node(state: DDIState) -> DDIState:
    actions = set(state.plan.get("actions", []))
    # health.kr enrich
    if "healthkr:drugA" in actions and state.drugA:
        cd = search_healthkr_drug_cd(state.drugA)
        if cd: state.healthA = fetch_healthkr_fields(cd).__dict__
    if "healthkr:drugB" in actions and state.drugB:
        cd = search_healthkr_drug_cd(state.drugB)
        if cd: state.healthB = fetch_healthkr_fields(cd).__dict__

    # ddinter pair
    if "ddinter:pair" in actions and state.drugA and state.drugB:
        row = parse_ddi2_pair(state.drugA, state.drugB)
        state.ddi = {
            "결과": row.결과, "사유": row.사유, "중증도": row.중증도,
            "메커니즘": row.메커니즘, "근거문헌": row.근거문헌,
        }
    return state

# ---------------- Utils ----------------

def _last_user(state: DDIState) -> str:
    return (state.messages[-1]["content"] if state.messages else state.query).strip()

# ---------------- Nodes ----------------

def parse_query_node(state: DDIState) -> DDIState:
    q = _last_user(state)
    # very light split by common separators
    import re
    toks = [t.strip() for t in re.split(r"[,+/]|\s+vs\s+|\s+and\s+|\s*&\s+", q) if t.strip()]
    if len(toks) >= 2:
        state.drugA, state.drugB = toks[0], toks[1]
    else:
        state.drugA = q
        state.drugB = ""
    state.query = q
    return state


async def _to_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

# Example in a node:
def _fetch_health_fields(drug_name: str):
    cd = search_healthkr_drug_cd(drug_name)        # uses requests
    return fetch_healthkr_fields(cd).__dict__ if cd else {}

async def healthkr_enrich_node(state):
    if state.drugA:
        state.healthA = await _to_thread(_fetch_health_fields, state.drugA)
    if state.drugB:
        state.healthB = await _to_thread(_fetch_health_fields, state.drugB)
    return state


async def _parse_ddi_pair(a, b):
    row = parse_ddi2_pair(a, b)
    return {
        "결과": row.결과, "사유": row.사유, "중증도": row.중증도,
        "메커니즘": row.메커니즘, "근거문헌": row.근거문헌,
    }

async def ddinter_node(state: DDIState) -> DDIState:
    if state.drugA and state.drugB:
        state.ddi = await _to_thread(_parse_ddi_pair, state.drugA, state.drugB)
    return state


async def graph_upsert_node(state: DDIState, neo: GraphStore) -> DDIState:
    # Ensure constraints once (off-thread)
    await _to_thread(neo.ensure_constraints_once)

    # Upserts off-thread
    if state.drugA:
        await _to_thread(neo.upsert_drug, state.drugA, state.healthA or {})
    if state.drugB:
        await _to_thread(neo.upsert_drug, state.drugB, state.healthB or {})
    if state.drugA and state.drugB and state.ddi:
        await _to_thread(
            neo.upsert_interaction,
            state.drugA, state.drugB,
            state.ddi.get("중증도",""), "interaction",
            state.ddi.get("메커니즘",[]), state.ddi.get("근거문헌",[])
        )
    return state


def answer_node(state: DDIState) -> DDIState:
    if state.drugA and not state.drugB:
        a = state.drugA
        h = state.healthA or {}
        eff = h.get("효능효과_md") or "- (수집된 효능/효과 없음)"
        state.answer = (
            f"**{a}** 기본정보 (health.kr 파싱)\n\n"
            f"- 식약처 분류: {h.get('식약처분류','')}\n"
            f"- 성분: {h.get('성분1','')}, {h.get('성분2','')}, {h.get('성분3','')}\n"
            f"- 대상: {h.get('대상','')}\n\n"
            f"**효능/효과**\n{eff}\n\n"
            "⚠️ 본 정보는 참고용입니다. 개인 복약은 반드시 전문가 상담이 필요합니다."
        )
        return state

    if state.drugA and state.drugB:
        a,b = state.drugA, state.drugB
        d = state.ddi or {}
        mechs = ", ".join(d.get("메커니즘") or []) or "(명시 없음)"
        refs = d.get("근거문헌") or []
        ref_md = "\n".join([f"  - [{r.get('title','ref')}]({r.get('url','')})" for r in refs])
        state.answer = (
            f"**{a} + {b} 상호작용**\n\n"
            f"- 결과: {d.get('결과','')}\n"
            f"- 중증도: {d.get('중증도','(알수없음)')}\n"
            f"- 사유: {d.get('사유','')}\n"
            f"- 메커니즘: {mechs}\n\n"
            f"**근거**\n{ref_md or '- (링크 없음)'}\n\n"
            "⚠️ 본 정보는 참고용입니다. 개인 복약은 반드시 전문가 상담이 필요합니다."
        )
        return state

    state.answer = "질문을 이해하지 못했습니다. 제품명을 입력해 주세요."
    return state

async def llm_search_node(state: DDIState) -> DDIState:
    q = (state.plan.get("web_search_query") or "").strip()
    if not q:
        if state.drugA and state.drugB:
            q = f"{state.drugA} {state.drugB} 상호작용 근거 site:health.kr OR site:scbdd.com"
        elif state.drugA:
            q = f"{state.drugA} 효능 효과 성분 site:health.kr"
        else:
            q = state.query
    try:
        state.web_search_notes = await _to_thread(llm_search_then_summarize, q) or ""
    except Exception:
        state.web_search_notes = ""
    return state

def optional_llm_search_node(state: DDIState) -> DDIState:
    q = (state.plan or {}).get("web_search_query","").strip()
    if q:
        try:
            state.web_search_notes = llm_search_then_summarize(q)
        except Exception:
            state.web_search_notes = ""
    return state

def db_sync_node(state: DDIState):
    if not USE_LOCAL_DB:
        return state
    df = load_db()

    # Single-drug enrichment → write 성분/분류/효능/대상
    if state.drugA and not state.drugB and state.healthA:
        h = state.healthA
        fields = {
            '성분1': h.get('성분1',''),
            '성분2': h.get('성분2',''),
            '성분3': h.get('성분3',''),
            '식약처분류': h.get('식약처분류',''),
            '효능/효과1': (h.get('효능효과_md','').splitlines()[0].removeprefix("- ").strip()
                           if h.get('효능효과_md') else ''),
            '효능/효과2': (h.get('효능효과_md','').splitlines()[1].removeprefix("- ").strip()
                           if h.get('효능효과_md') and len(h.get('효능효과_md').splitlines())>1 else ''),
            '대상': h.get('대상','')
        }
        df = upsert_single(df, state.drugA, fields)

    # Two-drug enrichment → write both singles (if present) + pair row
    if state.drugA and state.drugB:
        if state.healthA:
            df = upsert_single(df, state.drugA, {
                '성분1': state.healthA.get('성분1',''),
                '성분2': state.healthA.get('성분2',''),
                '성분3': state.healthA.get('성분3',''),
                '식약처분류': state.healthA.get('식약처분류',''),
                '효능/효과1': (state.healthA.get('효능효과_md','').splitlines()[0].removeprefix("- ").strip()
                               if state.healthA.get('효능효과_md') else ''),
                '효능/효과2': (state.healthA.get('효능효과_md','').splitlines()[1].removeprefix("- ").strip()
                               if state.healthA.get('효능효과_md') and len(state.healthA.get('효능효과_md').splitlines())>1 else ''),
                '대상': state.healthA.get('대상','')
            })
        if state.healthB:
            df = upsert_single(df, state.drugB, {
                '성분1': state.healthB.get('성분1',''),
                '성분2': state.healthB.get('성분2',''),
                '성분3': state.healthB.get('성분3',''),
                '식약처분류': state.healthB.get('식약처분류',''),
                '효능/효과1': (state.healthB.get('효능효과_md','').splitlines()[0].removeprefix("- ").strip()
                               if state.healthB.get('효능효과_md') else ''),
                '효능/효과2': (state.healthB.get('효능효과_md','').splitlines()[1].removeprefix("- ").strip()
                               if state.healthB.get('효능효과_md') and len(state.healthB.get('효능효과_md').splitlines())>1 else ''),
                '대상': state.healthB.get('대상','')
            })
        # Pair row from ddinter
        if state.ddi:
            df = upsert_pair(df, state.drugA, state.drugB,
                             결과=state.ddi.get('결과',''),
                             사유=state.ddi.get('사유',''))

    save_db(df)
    return state


def check_db_node(state: DDIState) -> DDIState:
    # we’ll just compute flags and stuff them into plan for routing
    if not USE_LOCAL_DB:
        state.plan["need_llm_search"] = True
        return state

    df = load_db()

    need = False
    if state.drugA and not state.drugB:
        need = not has_single(df, state.drugA)
    elif state.drugA and state.drugB:
        # prefer pair existence; if pair missing, we’ll search
        need = not has_pair(df, state.drugA, state.drugB)
        # (you could also require singles to exist—optional)
    else:
        need = True  # nothing parsed → be safe and search

    state.plan["need_llm_search"] = bool(need and USE_LLM_SEARCH)
    return state

def route_after_check(state: DDIState) -> str:
    return "search" if state.plan.get("need_llm_search") else "skip"


# ---------------- Build ----------------

def build_app():
    g = StateGraph(DDIState)
    neo = GraphStore()   

    g.add_node("parse_query", parse_query_node)
    g.add_node("llm_plan", llm_plan_node)
    g.add_node("check_db", check_db_node)                 # NEW
    g.add_node("llm_search", llm_search_node)            # NEW (optional)
    g.add_node("selective_retrieval", selective_retrieval_node)
    g.add_node("healthkr_enrich", healthkr_enrich_node)   # async
    g.add_node("ddinter", ddinter_node)                   # async
    g.add_node("db_sync", db_sync_node)                  # already created earlier
    g.add_node("graph_upsert", lambda s: graph_upsert_node(s, neo))  # ok; node func is async now
    g.add_node("answer", answer_node)

    g.set_entry_point("parse_query")
    g.add_edge("parse_query", "llm_plan")
    g.add_edge("llm_plan", "check_db")

    # Conditional route:
    g.add_conditional_edges("check_db", route_after_check, {
    "search": "llm_search",
    "skip":   "healthkr_enrich"
    })
    # If we did LLM search, continue to normal retrieval
    g.add_edge("llm_search", "healthkr_enrich")
    g.add_edge("healthkr_enrich", "ddinter")
    g.add_edge("ddinter", "db_sync")

    # Persist to CSV, then to graph, then answer
    g.add_edge("db_sync", "graph_upsert")
    g.add_edge("graph_upsert", "answer")

    return g.compile(checkpointer=MemorySaver())

def app():
    return build_app()

if __name__ == "__main__":
    app = build_app(NeoCfg())
    out = app.invoke({
        "messages": [{"role":"user","content":"크레스토정 10mg, 클라리스로마이신"}]
    })
    print(out["answer"])  # note: LangGraph returns dict-like state
