from langgraph.graph import StateGraph
from my_agent.state import QueryState
from my_agent.nodes import parse_and_canonicalize_node, db_or_web_lookup_node, store_result_and_init_node

graph = StateGraph(QueryState)
graph.add_node("parse_canonicalize", parse_and_canonicalize_node)
graph.add_node("lookup", db_or_web_lookup_node)
graph.add_node("store_and_init", store_result_and_init_node)
graph.set_entry_point("parse_canonicalize")
graph.add_edge("parse_canonicalize", "lookup")
graph.add_edge("lookup", "store_and_init")
graph.set_finish_point("store_and_init")