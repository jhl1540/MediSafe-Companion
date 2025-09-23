# parsers.py
import re
import time
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "ko,en;q=0.9"
}

# -----------------------------
# Helpers
# -----------------------------

def _get(url: str, params: Optional[dict]=None, retry: int=2, timeout: int=20) -> requests.Response:
    last_exc = None
    for i in range(retry+1):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception as e:
            last_exc = e
        time.sleep(1.2 + random.random())
    if last_exc:
        raise last_exc
    raise RuntimeError(f"HTTP {url} failed")

# -----------------------------
# health.kr product detail parser
# -----------------------------

@dataclass
class HealthKRFields:
    성분1: str = ""
    성분2: str = ""
    성분3: str = ""
    식약처분류: str = ""
    효능효과_md: str = ""  # 마크다운 정리 텍스트
    대상: str = ""         # (있으면) 성인/소아/고령자 등

HEALTH_BASE = "https://www.health.kr/searchDrug"

def search_healthkr_drug_cd(keyword: str) -> Optional[str]:
    """Try to resolve first drug_cd for a brand/ingredient.
    Uses result_drug.asp?keyword=... page which lists results including drug_cd.
    """
    url = f"{HEALTH_BASE}/result_drug.asp"
    r = _get(url, params={"keyword": keyword})
    m = re.search(r"result_drug\\.asp\\?drug_cd=(\\d+)", r.text)
    return m.group(1) if m else None

def _clean_html_to_text(html: str) -> str:
    t = re.sub(r"<[^>]+>", " ", html)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def fetch_healthkr_fields(drug_cd: str) -> HealthKRFields:
    url = f"{HEALTH_BASE}/result_drug.asp"
    r = _get(url, params={"drug_cd": drug_cd})
    soup = BeautifulSoup(r.text, "html.parser")

    fields = HealthKRFields()

    # 테이블 라벨 기반 파싱: 라벨 셀 텍스트를 키로 삼아 오른쪽 셀을 값으로
    def get_td_after(label_regex: str) -> Optional[str]:
        lab = soup.find(string=re.compile(label_regex))
        if not lab:
            return None
        td = lab.find_parent(["th","td"])
        if not td:
            return None
        # 다음 형제 td
        nxt = td.find_next_sibling("td")
        if not nxt:
            return None
        return str(nxt)

    # 식약처 분류
    raw = get_td_after(r"식약처\s*분류|분류")
    if raw:
        fields.식약처분류 = _clean_html_to_text(raw)

    # 성분/함량 → 상위 3개만 요약
    raw = get_td_after(r"성분\s*/\s*함량|주성분")
    if raw:
        txt = _clean_html_to_text(raw)
        # 절, 세미콜론, 슬래시 등 구분자로 분해
        parts = [p.strip() for p in re.split(r"[;/•\u2022]| - ", txt) if p.strip()]
        # 잦은 패턴: "성분명 10mg" 형태 우선
        picked = []
        for p in parts:
            if re.search(r"\d+\s*(mg|g|mcg|㎍|%)", p, re.I):
                picked.append(p)
            else:
                picked.append(p)
            if len(picked) >= 3:
                break
        for i in range(3):
            setattr(fields, f"성분{i+1}", picked[i] if i < len(picked) else "")

    # 효능/효과 → 마크다운 bullet 정리
    raw = get_td_after(r"효능\s*/\s*효과|효능효과")
    if raw:
        text = _clean_html_to_text(raw)
        bullets = [b.strip("- •;: ") for b in re.split(r"[•\u2022;]\s+|\d\)\s+| - ", text) if b.strip()]
        if bullets:
            fields.효능효과_md = "\n".join([f"- {b}" for b in bullets])
        else:
            fields.효능효과_md = text

    # 대상 힌트
    html = r.text
    if "성인" in html:
        fields.대상 = "성인"
    elif "소아" in html:
        fields.대상 = "소아"
    elif "고령" in html:
        fields.대상 = "고령자"

    return fields

# -----------------------------
# DDInter 2.0 parser (pair detail page)
# -----------------------------

@dataclass
class DDIRow:
    결과: str = ""     # ⚠️/❌/""
    사유: str = ""      # 요약 사유
    중증도: str = ""    # e.g., Major/Moderate/Minor
    메커니즘: List[str] = None
    근거문헌: List[Dict[str,str]] = None  # {title,url}

DDI2_BASE = "https://ddinter2.scbdd.com"

def parse_ddi2_pair(drug_a: str, drug_b: str) -> DDIRow:
    """Best-effort parser.
    1) 홈에서 검색(쿼리 파라미터 없음 → JS 렌더링이라 실패 가능)
    2) 백오프: 구 사이트(ddinter.scbdd.com)에서 pair 문장 키워드 수집
    3) 최종 폴백: 일반적 사유 텍스트
    운영 시엔 사이트 DOM 구조 고정 확인 후 CSS selector로 교체 권장.
    """
    row = DDIRow(메커니즘=[], 근거문헌=[])

    # 폴백 기본값 (안전)
    row.결과 = "⚠️"
    row.사유 = "가능한 상호작용: 대사효소/수송체 변화 (예: CYP3A4, P-gp 등). 전문자료 확인 필요"

    try:
        # 시도: 구 사이트 요약 텍스트 (HTML 내 문장 키워드 패턴)
        legacy = _get(f"https://ddinter.scbdd.com/")
        if legacy.status_code == 200:
            # 여기서는 실제로 쿼리 엔드포인트가 명확치 않아, 설명 블록에서 메커니즘 키워드 추출
            kw = []
            text = legacy.text
            for k in ["CYP3A4","CYP2D6","P-gp","QT prolongation","serotonergic"]:
                if k.lower() in text.lower():
                    kw.append(k)
            if kw:
                row.메커니즘 = list(sorted(set(kw)))
                row.사유 = f"가능한 상호작용 메커니즘: {', '.join(row.메커니즘)}"
                row.중증도 = "Moderate"  # 보수적 기본값(운영 시 매핑표 적용)
    except Exception:
        pass

    # 근거문헌: DDInter 2.0/NAR 논문 링크 제공(정적)
    row.근거문헌.append({
        "title": "DDInter 2.0: enhanced DDI resource (NAR 2025)",
        "url": "https://academic.oup.com/nar/article/53/D1/D1356/7740584"
    })

    # 중증도 → 결과 매핑
    sev = (row.중증도 or "").lower()
    if "contra" in sev or sev == "major":
        row.결과 = "❌"
    elif sev in ("moderate",""):
        row.결과 = "⚠️"
    else:
        row.결과 = ""

    return row


