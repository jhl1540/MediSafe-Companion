import streamlit as st
import pandas as pd
import os

CSV_PATH = os.getenv("LOCAL_CSV", "완제의약품_허가_상세_2015-2024_통합.csv")

_df_cache = None
def _ensure_df():
    global _df_cache
    if _df_cache is None:
        try:
            _df_cache = pd.read_csv(CSV_PATH)
        except Exception:
            _df_cache = pd.DataFrame(columns=["ITEM_NAME","INGREDIENT","WARN_URL"])
    return _df_cache

def fuzzy_find(name: str, topn: int = 3):
    df = _ensure_df()
    if df.empty:
        return []
    n = (name or "").strip().lower()
    hits = df[df["ITEM_NAME"].str.lower().str.contains(n, na=False)].copy()
    hits = hits.head(topn)
    return hits.to_dict("records")

def render_db_info(drug_name: str):
    rows = fuzzy_find(drug_name, topn=1)
    if not rows:
        st.caption("해당 약물의 공공DB 레코드를 찾지 못했습니다.")
        return
    r = rows[0]
    st.markdown(f"**5. 주성분 표시:** {r.get('INGREDIENT','-')}")
    url = r.get("WARN_URL","")
    if isinstance(url, str) and url.strip():
        st.markdown(f"**6. 주의사항('15~'24 공공데이터 기준):**  \n🔗 [열기]({url}) | DB 등록명: {r.get('ITEM_NAME','')}")
    else:
        st.markdown("**6. 주의사항:** 등록된 링크가 없습니다.")
