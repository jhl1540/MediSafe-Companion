from __future__ import annotations
from .orchestrator import StateGraph, END
from .state import DDIState
from .nodes import (
    n0_kr_brand_fetcher,
    n1_query_normalizer,
    n2_embed_and_store,
    n3_web_updater,
    n4_ddi_ranker,
    n5_alternative_finder,
    n6_response_generator,
    n7_userdb_csv_writer,
)


def make_graph(_config=None):
    workflow = StateGraph(DDIState)
    workflow.add_node("N0_KR_BrandFetcher", n0_kr_brand_fetcher)
    workflow.add_node("N1_QueryNormalizer", n1_query_normalizer)
    workflow.add_node("N2_EmbedStore", n2_embed_and_store)
    workflow.add_node("N3_WebUpdater", n3_web_updater)
    workflow.add_node("N4_DDIRanker", n4_ddi_ranker)
    workflow.add_node("N5_AltFinder", n5_alternative_finder)
    workflow.add_node("N6_Response", n6_response_generator)
    workflow.add_node("N7_UserDB", n7_userdb_csv_writer)


    workflow.set_entry_point("N0_KR_BrandFetcher")
    workflow.add_edge("N0_KR_BrandFetcher", "N1_QueryNormalizer")
    workflow.add_edge("N1_QueryNormalizer", "N2_EmbedStore")
    workflow.add_edge("N2_EmbedStore", "N3_WebUpdater")
    workflow.add_edge("N3_WebUpdater", "N4_DDIRanker")
    workflow.add_edge("N4_DDIRanker", "N5_AltFinder")
    workflow.add_edge("N5_AltFinder", "N6_Response")
    workflow.add_edge("N6_Response", "N7_UserDB")
    workflow.add_edge("N7_UserDB", END)
    

    return workflow.compile()


# Convenience export for direct imports
graph = make_graph()


if __name__ == "__main__":
    # Manual quick run
    g = graph
    for q in ["부루펜정", "Tylenol", "Ethanol", "UnknownDrugX"]:
        print("======================")
        print("INPUT:", q)
        state = g.invoke({"user_query": q})
        print(state.get("response", "<no response>"))