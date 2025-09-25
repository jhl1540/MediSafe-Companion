# -*- coding: utf-8 -*-
"""
db_utils.py — Streamlit + RapidFuzz 기반 CSV(DB) 조회 유틸
- 원본 CSV는 그대로 유지합니다.
- (선택) 경량 CSV가 있으면 우선 사용하고, 없으면 원본 CSV를 사용합니다.
- ITEM_NAME+MAIN_ITEM_INGR(제품명+주성분) 기반의 부분일치/오타허용 검색, 5. 주성분 표시, 6. 주의사항(🔗) 렌더링.
"""
from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple

import pandas as pd
from rapidfuzz import process, fuzz
import streamlit as st

# === 설정(필요 시 변경) ===
PRIMARY_DB = "완제의약품_허가_상세_2015-2024_통합.csv"   # 원본 CSV(유지)
MINIMAL_DB = "med_db_minimal.csv"                         # (선택) 경량 3컬럼 CSV

# 컬럼 매핑(업로드 CSV 기준)
COL_ITEM = "ITEM_NAME"        # 제품명
COL_ING  = "MAIN_ITEM_INGR"   # 5. 주성분 표시(텍스트)
COL_WARN = "NB_DOC_ID"        # 6. 주의사항(URL)

# [변경 포인트 ★] 컷오프 한 곳에서만 조정하게 상수로 관리(0~100, 낮을수록 관대)
DEFAULT_CUTOFF = 50

def _normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # 공백·괄호·일부 기호 제거 + 소문자화(한글은 영향 없음)
    return re.sub(r"[\s\(\)\[\]{}·•\-\_/.,+]", "", s).lower()

@st.cache_data(show_spinner=False)
def load_db() -> pd.DataFrame:
    """경량 DB가 있으면 먼저 사용, 없으면 원본 CSV 사용."""
    for path in (MINIMAL_DB, PRIMARY_DB):
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
            for c in (COL_ITEM, COL_ING, COL_WARN):
                if c not in df.columns:
                    df[c] = ""
            # [변경 포인트 ★] 검색 키를 "제품명 + 주성분"으로 구성 → 제품명·주성분 둘 다로 검색 가능
            df["_key"] = (df[COL_ITEM].fillna("") + " " + df[COL_ING].fillna("")).map(_normalize)
            return df[[COL_ITEM, COL_ING, COL_WARN, "_key"]]
        except Exception:
            continue
    return pd.DataFrame(columns=[COL_ITEM, COL_ING, COL_WARN, "_key"])

def fuzzy_find(item_name: str, topn: int = 1, cutoff: int = DEFAULT_CUTOFF) -> List[Dict[str, Any]]:
    """
    입력 문자열과 ITEM_NAME+MAIN_ITEM_INGR 유사도 매칭.
    - 우선: '부분 문자열' 직접 포함(q ⊂ 후보) ⇒ 점수 100으로 채택(짧은 브랜드명 대응)
    - 보조: RapidFuzz partial_ratio(짧은↔긴 문자열에 강함)
    - cutoff: 0~100, 높을수록 엄격
    """
    df = load_db()
    if df.empty or not item_name:
        return []
    q = _normalize(item_name)
    cands = df["_key"].tolist()

    hits: List[Tuple[int, int]] = []

    # 0) 서브스트링 직매칭(예: "아토젯" ∈ "아토젯정1010밀리그램 …")
    if q:
        direct = [(100, i) for i, c in enumerate(cands) if q in c]
        hits.extend(direct[:topn])  # 상위 일부만

    # 1) 부족하면 퍼지(부분비교)로 보충
    remain = max(0, topn - len(hits))
    if remain > 0:
        extra = process.extract(
            q, cands,
            scorer=fuzz.partial_ratio,       # 짧은 입력에 강함
            limit=remain,
            score_cutoff=cutoff              # [변경 포인트 ★] cutoff 적용 위치
        )
        hits.extend([(score, idx) for _, score, idx in extra])

    # 2) 결과 구성(중복 제거 + 점수순)
    results: List[Dict[str, Any]] = []
    seen = set()
    for score, idx in sorted(hits, key=lambda x: -x[0]):
        if idx in seen:
            continue
        row = df.iloc[idx]
        results.append({
            "ITEM_NAME": row[COL_ITEM],
            "INGREDIENT": row[COL_ING],
            "WARN_URL": row[COL_WARN],
            "SCORE": score
        })
        seen.add(idx)
        if len(results) >= topn:
            break
    return results

def render_db_info(user_input: str, *, show_candidates: int = 1, cutoff: int = DEFAULT_CUTOFF) -> None:
    """
    약 카드 하단에 섹션 5, 6을 출력.
    - show_candidates>1 이면 상위 후보를 표로 함께 보여줌(옵션).
    - cutoff은 fuzzy_find로 그대로 전달(0~100, 높을수록 엄격).
    """
    try:
        hits = fuzzy_find(user_input, topn=show_candidates, cutoff=cutoff)  # [변경 포인트 ★] cutoff 전달
        best = hits[0] if hits else None

        # 5. 주성분 표시
        ing = (best["INGREDIENT"].strip() if best and isinstance(best.get("INGREDIENT",""), str) else "")
        if ing:
            st.markdown("**5. 주성분 표시**: " + ing)
        else:
            st.markdown("**5. 주성분 표시**: '15년~'24년 공공데이터 미 등록으로 세부 정보 확인 불가")

        # 6. 주의사항 라벨(윗줄) + 실제 링크/DB 등록명(아랫줄)
        st.markdown("**6. 주의사항('15년~'24년 공공데이터 등록 기준):**")
        url = (best.get("WARN_URL","") if best else "")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            st.markdown(f"[🔗 열기]({url}) | **DB 등록명:** {best.get('ITEM_NAME','')}")
        else:
            st.markdown("'15년~'24년 공공데이터 미 등록으로 세부 정보 확인 불가")

        # (옵션) 후보 표
        if show_candidates > 1 and hits:
            import pandas as _pd
            _tab = _pd.DataFrame(
                [{"제품명": h["ITEM_NAME"], "유사도": h["SCORE"], "주의사항URL": h["WARN_URL"]} for h in hits]
            )
            st.dataframe(_tab, hide_index=True, use_container_width=True)

    except Exception:
        st.markdown("**5. 주성분 표시**: '15년~'24년 공공데이터 미 등록으로 세부 정보 확인 불가")
        st.markdown("**6. 주의사항('15년~'24년 공공데이터 등록 기준):**")
        st.markdown("'15년~'24년 공공데이터 미 등록으로 세부 정보 확인 불가")
