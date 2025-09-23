from typing import TypedDict, Optional, List

class QueryState(TypedDict):
    user_input1: str         # User input for 제품명1 (may be substring/abbreviation)
    user_input2: Optional[str]  # User input for 제품명2 (may be None or blank)
    drug1: Optional[str]     # Canonicalized drug name for 제품명1 (matched/formatted)
    drug2: Optional[str]     # Canonicalized drug name for 제품명2
    result: Optional[List[dict]]
    source: Optional[str]    # "db" or "web"
    error: Optional[str]