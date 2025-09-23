\
from langchain_core.tools import tool
from .db_cache import lookup_single, lookup_pair, upsert_single, upsert_pair
from .neo4j_tool import Neo4jClient

neo4j_client = Neo4jClient()

@tool("db_lookup_single", return_direct=False)
def db_lookup_single(drug: str):
    """Look up single-drug info from CSV DB. Returns None if not found."""
    return lookup_single(drug)

@tool("db_lookup_pair", return_direct=False)
def db_lookup_pair(drug_a: str, drug_b: str):
    """Look up pairwise interaction info from CSV DB. Returns None if not found."""
    return lookup_pair(drug_a, drug_b)

@tool("db_upsert_single", return_direct=False)
def db_upsert_single(drug: str, info: dict):
    """Upsert single-drug info into CSV DB."""
    upsert_single(drug, info)
    return {"status": "ok"}

@tool("db_upsert_pair", return_direct=False)
def db_upsert_pair(drug_a: str, drug_b: str, info: dict):
    """Upsert pair info into CSV DB."""
    upsert_pair(drug_a, drug_b, info)
    return {"status": "ok"}

@tool("neo4j_single", return_direct=False)
def neo4j_single(drug: str):
    """Query Neo4j for single drug info following the CSV schema."""
    return neo4j_client.single_info(drug)

@tool("neo4j_pair", return_direct=False)
def neo4j_pair(drug_a: str, drug_b: str):
    """Query Neo4j for pair interaction info following the CSV schema."""
    return neo4j_client.pair_info(drug_a, drug_b)
