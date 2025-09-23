import os
import re
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
from filelock import FileLock
from .config import DB_CSV

COLUMN_ALIASES = {
    "drug": ["drug", "drug_name", "제품명", "약품명"],
    "component": ["component", "components", "성분", "성분1", "성분_리스트"],
    "partner": ["partner", "상대약물", "상호작용상대", "제품명2", "상대"],
    "interaction": ["interaction", "상호작용", "설명", "사유"],
    "severity": ["severity", "등급", "결과"],
    "source": ["source", "출처"],
    "updated_at": ["updated_at", "업데이트", "수정일"],
    "confidence": ["confidence"],
    "evidence": ["evidence", "근거"],
}

REQUIRED_CANONICAL = [
    "drug","component","partner","interaction","severity",
    "source","updated_at","confidence","evidence",
]

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a.lower() in lower_cols:
                mapping[lower_cols[a.lower()]] = canon
                break
    df = df.rename(columns=mapping)
    for c in REQUIRED_CANONICAL:
        if c not in df.columns:
            df[c] = None
    return df[REQUIRED_CANONICAL]

def read_db() -> pd.DataFrame:
    if not os.path.exists(DB_CSV):
        return pd.DataFrame(columns=REQUIRED_CANONICAL)
    with FileLock(DB_CSV + ".lock"):
        df = pd.read_csv(DB_CSV)
    df = _canonicalize_columns(df)
    return df

def write_db(df: pd.DataFrame) -> None:
    df = _canonicalize_columns(df)
    with FileLock(DB_CSV + ".lock"):
        df.to_csv(DB_CSV, index=False)

def search_db_for_drug(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not name:
        return df.iloc[0:0]
    pat = re.escape(name.lower())
    mask = df["drug"].fillna("").str.lower().str.contains(pat)
    mask_partner = df["partner"].fillna("").str.lower().str.contains(pat)
    return df[mask | mask_partner]

def get_monograph(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    subset = search_db_for_drug(df, name)
    comps = sorted({c for c in subset["component"].dropna().astype(str).tolist() if c})
    interactions = subset.to_dict(orient="records")
    return {"drug": name, "components": comps, "interactions": interactions}

def upsert_interaction(
    df: pd.DataFrame,
    drug: str,
    component: str,
    partner: str,
    interaction: str,
    severity: str = "",
    source: str = "",
    confidence: float = 0.5,
    evidence: str = "",
) -> pd.DataFrame:
    now = datetime.utcnow().isoformat()
    row = {
        "drug": drug,
        "component": component,
        "partner": partner,
        "interaction": interaction,
        "severity": severity,
        "source": source,
        "updated_at": now,
        "confidence": confidence,
        "evidence": evidence,
    }
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)