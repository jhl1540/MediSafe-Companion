\
import os
import pandas as pd

CSV_PATH = os.getenv("DB_CSV", "/mnt/data/DB.csv")

EXPECTED_COLS = [
    "제품명","성분1","성분2","성분3","식약처분류",
    "효능/효과1","효능/효과2","대상","결과",
    "제품명2","사유"
]

def _read_csv():
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            df = pd.read_csv(CSV_PATH, encoding=enc)
            if "제품명2" not in df.columns and "제품명.1" in df.columns:
                df = df.rename(columns={"제품명.1": "제품명2"})
            for c in EXPECTED_COLS:
                if c not in df.columns:
                    df[c] = ""
            return df[EXPECTED_COLS]
        except Exception:
            continue
    return pd.DataFrame(columns=EXPECTED_COLS)

def _write_csv(df: pd.DataFrame):
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[EXPECTED_COLS]
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def lookup_single(drug_name: str):
    df = _read_csv()
    if df.empty:
        return None
    mask = df["제품명"].astype(str).str.lower().str.strip() == _norm(drug_name)
    hits = df[mask]
    if hits.empty:
        return None
    return hits.iloc[0][[
        "제품명","성분1","성분2","성분3","식약처분류","효능/효과1","효능/효과2","대상","결과"
    ]].fillna("").to_dict()

def lookup_pair(drug_a: str, drug_b: str):
    df = _read_csv()
    if df.empty:
        return None
    a, b = _norm(drug_a), _norm(drug_b)
    mask1 = (df["제품명"].astype(str).str.lower().str.strip() == a) & (df["제품명2"].astype(str).str.lower().str.strip() == b)
    mask2 = (df["제품명"].astype(str).str.lower().str.strip() == b) & (df["제품명2"].astype(str).str.lower().str.strip() == a)
    hits = df[mask1 | mask2]
    if hits.empty:
        return None
    row = hits.iloc[0]
    return {
        "제품명1": row.get("제품명",""),
        "제품명2": row.get("제품명2",""),
        "사유": row.get("사유",""),
        "결과": row.get("결과",""),
        "성분1": row.get("성분1",""),
        "성분2": row.get("성분2",""),
        "성분3": row.get("성분3",""),
        "식약처분류": row.get("식약처분류",""),
        "효능/효과1": row.get("효능/효과1",""),
        "효능/효과2": row.get("효능/효과2",""),
        "대상": row.get("대상",""),
    }

def upsert_single(drug_name: str, info: dict):
    df = _read_csv()
    mask = df["제품명"].astype(str).str.lower().str.strip() == _norm(drug_name)
    row = {
        "제품명": drug_name,
        "성분1": info.get("성분1",""),
        "성분2": info.get("성분2",""),
        "성분3": info.get("성분3",""),
        "식약처분류": info.get("식약처분류",""),
        "효능/효과1": info.get("효능/효과1",""),
        "효능/효과2": info.get("효능/효과2",""),
        "대상": info.get("대상",""),
        "결과": info.get("결과",""),
        "제품명2": info.get("제품명2",""),
        "사유": info.get("사유",""),
    }
    if mask.any():
        df.loc[mask, row.keys()] = pd.Series(row)
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _write_csv(df)

def upsert_pair(drug_a: str, drug_b: str, info: dict):
    df = _read_csv()
    a, b = _norm(drug_a), _norm(drug_b)
    mask1 = (df["제품명"].astype(str).str.lower().str.strip() == a) & (df["제품명2"].astype(str).str.lower().str.strip() == b)
    mask2 = (df["제품명"].astype(str).str.lower().str.strip() == b) & (df["제품명2"].astype(str).str.lower().str.strip() == a)
    row = {
        "제품명": drug_a,
        "제품명2": drug_b,
        "사유": info.get("사유",""),
        "결과": info.get("결과",""),
        "성분1": info.get("성분1",""),
        "성분2": info.get("성분2",""),
        "성분3": info.get("성분3",""),
        "식약처분류": info.get("식약처분류",""),
        "효능/효과1": info.get("효능/효과1",""),
        "효능/효과2": info.get("효능/효과2",""),
        "대상": info.get("대상",""),
    }
    if mask1.any():
        df.loc[mask1, row.keys()] = pd.Series(row)
    elif mask2.any():
        df.loc[mask2, row.keys()] = pd.Series(row)
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _write_csv(df)
