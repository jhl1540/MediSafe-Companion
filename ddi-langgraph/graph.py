\
import os, re, json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from tools import (
    db_lookup_single, db_lookup_pair, db_upsert_single, db_upsert_pair,
    neo4j_single, neo4j_pair
)

SYSTEM = (
    "You are a DDI assistant for Korean drug queries. "
    "If the user asks about ONE drug, retrieve 제품명/성분1/성분2/성분3/식약처분류/효능/효과1/효능/효과2/대상/결과. "
    "If TWO drugs, also include 제품명2 and 사유, and set 결과 to the interaction severity/result. "
    "Always check the CSV DB first; if not found, query Neo4j; then upsert into CSV. "
    "Answer succinctly in Korean with a brief disclaimer: '의료 조언이 아닙니다.'"
)

class S(TypedDict):
    messages: List[Dict[str, Any]]

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

def parse_drugs(text: str):
    tokens = re.split(r"[\\s,;/＋+과와및&]|그리고|/|\\\\|,|;", text.strip())
    toks = [t for t in [tok.strip() for tok in tokens] if t]
    if len(toks) >= 2:
        return toks[0], toks[1]
    return (toks[0],) if toks else tuple()

def agent(state: S):
    last = state["messages"][-1]["content"]
    drugs = parse_drugs(last)

    msgs = [{"role":"system","content":SYSTEM}] + state["messages"]

    if len(drugs) == 1:
        toolres = db_lookup_single.invoke({"drug": drugs[0]})
        if toolres:
            final = llm.invoke(msgs + [AIMessage(content=f"DB 결과:\\n{json.dumps(toolres,ensure_ascii=False)}")])
            return {"messages": state["messages"] + [final.dict()]}
        fetched = neo4j_single.invoke({"drug": drugs[0]})
        if fetched:
            db_upsert_single.invoke({"drug": drugs[0], "info": fetched})
            final = llm.invoke(msgs + [AIMessage(content=f"그래프 결과:\\n{json.dumps(fetched,ensure_ascii=False)}")])
            return {"messages": state["messages"] + [final.dict()]}
        final = llm.invoke(msgs + [AIMessage(content="DB와 그래프에서 결과를 찾지 못했습니다.")])
        return {"messages": state["messages"] + [final.dict()]}

    elif len(drugs) >= 2:
        a, b = drugs[0], drugs[1]
        toolres = db_lookup_pair.invoke({"drug_a": a, "drug_b": b})
        if toolres:
            final = llm.invoke(msgs + [AIMessage(content=f"DB 결과:\\n{json.dumps(toolres,ensure_ascii=False)}")])
            return {"messages": state["messages"] + [final.dict()]}
        fetched = neo4j_pair.invoke({"drug_a": a, "drug_b": b})
        if fetched:
            db_upsert_pair.invoke({"drug_a": a, "drug_b": b, "info": fetched})
            final = llm.invoke(msgs + [AIMessage(content=f"그래프 결과:\\n{json.dumps(fetched,ensure_ascii=False)}")])
            return {"messages": state["messages"] + [final.dict()]}
        final = llm.invoke(msgs + [AIMessage(content="DB와 그래프에서 결과를 찾지 못했습니다.")])
        return {"messages": state["messages"] + [final.dict()]}

    else:
        final = llm.invoke(msgs + [AIMessage(content="약물명을 1개 또는 2개 입력해 주세요.")])
        return {"messages": state["messages"] + [final.dict()]}

graph = StateGraph(S)
graph.add_node("agent", agent)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
# memory = MemorySaver()
# app = graph.compile(checkpointer=memory)
app = graph.compile()