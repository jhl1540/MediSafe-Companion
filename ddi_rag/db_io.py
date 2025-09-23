# db_io.py
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

DB_PATH = Path(os.getenv("DB_PATH", "data/DB.csv"))
DB_COLUMNS = ['제품명1','성분1','성분2','성분3','식약처분류','효능/효과1','효능/효과2','대상','결과','제품명2','사유']

def ensure_db_dir():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_db() -> pd.DataFrame:
    ensure_db_dir()
    if DB_PATH.exists():
        df = pd.read_csv(DB_PATH)
        # normalize columns if file created earlier
        for c in DB_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return df[DB_COLUMNS]
    return pd.DataFrame(columns=DB_COLUMNS)

def save_db(df: pd.DataFrame):
    ensure_db_dir()
    # keep column order
    df = df.reindex(columns=DB_COLUMNS, fill_value="")
    df.to_csv(DB_PATH, index=False)

def upsert_single(df: pd.DataFrame, 제품명1: str, fields: Dict[str, str]) -> pd.DataFrame:
    mask = (df['제품명1'].fillna('').str.lower() == 제품명1.lower()) & (df['제품명2'].fillna('') == '')
    if mask.any():
        for k,v in fields.items():
            if v not in (None, ""):
                df.loc[mask, k] = v
    else:
        row = {c:"" for c in DB_COLUMNS}
        row['제품명1'] = 제품명1
        row.update(fields or {})
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df

def upsert_pair(df: pd.DataFrame, 제품명1: str, 제품명2: str, 결과: str, 사유: str) -> pd.DataFrame:
    mask = (df['제품명1'].fillna('').str.lower()==제품명1.lower()) & \
           (df['제품명2'].fillna('').str.lower()==제품명2.lower())
    if mask.any():
        if 결과: df.loc[mask, '결과'] = 결과
        if 사유: df.loc[mask, '사유'] = 사유
    else:
        row = {c:"" for c in DB_COLUMNS}
        row['제품명1'] = 제품명1
        row['제품명2'] = 제품명2
        row['결과'] = 결과 or ""
        row['사유'] = 사유 or ""
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def has_single(df: pd.DataFrame, 제품명1: str) -> bool:
    mask = (df['제품명1'].fillna('').str.lower() == (제품명1 or '').lower()) & \
           (df['제품명2'].fillna('') == '')
    return bool(mask.any())

def has_pair(df: pd.DataFrame, a: str, b: str) -> bool:
    mask = (df['제품명1'].fillna('').str.lower() == (a or '').lower()) & \
           (df['제품명2'].fillna('').str.lower() == (b or '').lower())
    return bool(mask.any())