# langgraph_workflow.py
from typing import Dict, Any
from langgraph.graph import StateGraph
from langgraph.constants import START, END

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from prompt_templates import (
    single_drug_system, single_drug_user,
    interaction_system, interaction_user
)

# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────
def analyze_single(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    msgs = [
        SystemMessage(content=single_drug_system),
        HumanMessage(content=single_drug_user.format(drug=state["drug1"]))
    ]
    return {"result": llm.invoke(msgs).content}

def analyze_interaction(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    msgs = [
        SystemMessage(content=interaction_system),
        HumanMessage(content=interaction_user.format(
            drug1=state["drug1"], drug2=state.get("drug2", "")
        ))
    ]
    return {"result": llm.invoke(msgs).content}

# Router: decide next node name ("single" | "pair")
def route(state: Dict[str, Any]) -> str:
    return "pair" if state.get("drug2") else "single"

# ─────────────────────────────────────────────────────────────────────────────
# Graph builder (Option A)
# ─────────────────────────────────────────────────────────────────────────────
def build_graph():
    sg = StateGraph(dict)

    # register nodes
    sg.add_node("single", analyze_single)
    sg.add_node("pair",   analyze_interaction)

    # START → conditional route → target node
    sg.add_conditional_edges(
        START,
        route,
        {
            "single": "single",
            "pair":   "pair",
        },
    )

    # terminal edges
    sg.add_edge("single", END)
    sg.add_edge("pair",   END)

    return sg.compile()

# ─────────────────────────────────────────────────────────────────────────────
# Simple chunk indexer used by the "인덱스(텍스트)" 탭
# ─────────────────────────────────────────────────────────────────────────────
def index_text_chunk(state: Dict[str, Any]):
    """
    Simplified: create (Document)-[:HAS_CHUNK]->(Chunk) and link (Chunk)-[:MENTIONS]->(Drug)
    by naive token match against Drug.display_name (lowercased exact).
    """
    from neo4j_store import GraphStore

    text = state["text"]
    store = GraphStore()
    store.ensure_schema()

    # naive tokenization for demo purposes
    candidates = list({w.strip(".,;()[]") for w in text.split() if len(w) >= 3})

    cy = """
    MERGE (doc:Document {doc_id:$doc_id})
      ON CREATE SET doc.title=$title, doc.source_url=$source_url, doc.createdAt=datetime()
    MERGE (c:Chunk {chunk_id:$chunk_id})
      ON CREATE SET c.text=$text
    MERGE (doc)-[:HAS_CHUNK]->(c)
    WITH c, $cands AS cands
    UNWIND cands AS nm
    WITH c, toLower(nm) AS nm
    MATCH (d:Drug)
    WHERE d.name = nm OR toLower(d.display_name) = nm
    MERGE (c)-[:MENTIONS]->(d)
    RETURN count(*) AS linked
    """

    with store._driver.session(database=store._database) as s:
        s.run(
            cy,
            doc_id=state["doc_id"],
            chunk_id=state["chunk_id"],
            text=text,
            title=state.get("title"),
            source_url=state.get("source_url"),
            cands=candidates
        ).consume()

    store.close()
    return {"chunk_id": state["chunk_id"], "mentions_linked": True}
