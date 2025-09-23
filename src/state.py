from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict

# Severity order used for ranking
SEVERITY_ORDER = {"High": 3, "Moderate": 2, "Low": 1}


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
    # KR brand scraping additions
    brand_page_url: Optional[str]
    brand_scrape: Optional[Dict[str, Any]]
    csv_path: Optional[str]
    secondary_query: str                 # optional second brand
    brand1: Optional[str]
    brand2: Optional[str]
    brand1_page_url: Optional[str]
    brand2_page_url: Optional[str]
    brand1_scrape: Optional[Dict[str, Any]]
    brand2_scrape: Optional[Dict[str, Any]]
    csv_userdb_path: Optional[str]
    csv_brandlog_path: Optional[str]
