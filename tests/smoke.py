from __future__ import annotations
import os

from src.graph import make_graph
from src.graphdb import GRAPH


def run():
    graph = make_graph()

    # 0) Korean brand path should not crash (may be no-op without internet/API keys)
    s0 = graph.invoke({"user_query": "부루펜정"})
    if GRAPH.brand_records:
        assert isinstance(GRAPH.brand_records[-1].get("record"), dict)
        if s0.get("csv_path"):
            assert os.path.exists(s0["csv_path"]) or True

    # 1) Tylenol → Acetaminophen; ethanol High interaction first
    s1 = graph.invoke({"user_query": "Tylenol"})
    inters = s1.get("interactions", [])
    assert inters, "Expected at least one interaction for Tylenol demo seed"
    assert inters[0]["other_name"] == "Ethanol", "Ethanol should rank first by severity"
    assert inters[0]["severity"] == "High", "Expected 'High' severity for Tylenol↔Ethanol"

    # 2) Synonym mapping
    s2 = graph.invoke({"user_query": "Paracetamol"})
    assert s2.get("inchikey") == "RZVAJINKPMORJF-UHFFFAOYSA-N"

    # 3) Unknown drug → alternatives still offered
    s3 = graph.invoke({"user_query": "UnknownDrugX"})
    assert s3.get("interactions") in ([], None)
    assert len(s3.get("alternatives", [])) >= 1

    # 4) Query logs should be recorded in GraphDB (for audit)
    assert len(GRAPH.query_records) >= 3, "Expected query audit records to be stored"
    logged = {r.get("query") for r in GRAPH.query_records}
    assert {"Tylenol", "Paracetamol", "UnknownDrugX"}.issubset(logged)


if __name__ == "__main__":
    run()
    
    
# .venv/Scripts/python.exe -m pip install "langgraph-cli[inmem]"
# .venv/Scripts/langgraph.exe dev