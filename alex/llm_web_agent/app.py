# app.py
import re
import uuid
import streamlit as st
from langgraph_workflow import build_graph, index_text_chunk
from db_utils import render_db_info, fuzzy_find
from neo4j_store import GraphStore

def _first_row(hit):
    """fuzzy_find 결과에서 첫 행을 dict로 반환(없으면 {})."""
    if not hit:
        return {}
    row = hit[0]
    # pandas Series 지원
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    return dict(row)

def _pick(row: dict, *candidates):
    """row에서 후보 키들을 순서대로 탐색해서 첫 값을 반환, 없으면 ''. (대소문자/언더스코어 무시)"""
    if not row:
        return ""
    # 1) 정확 키
    for k in candidates:
        if k in row and row[k]:
            return row[k]
    # 2) case-insensitive + underscore-less 매칭
    norm = {str(k).lower().replace("_",""): v for k, v in row.items()}
    for k in candidates:
        kk = str(k).lower().replace("_","")
        if kk in norm and norm[kk]:
            return norm[kk]
    # 3) 부분 문자열 힌트(한국어 컬럼명 대응)
    keys = {str(k).lower(): k for k in row.keys()}
    for hint in candidates:
        h = str(hint).lower()
        for lk, orig in keys.items():
            if h in lk and row.get(orig):
                return row[orig]
    return ""

@st.cache_resource
def get_store():
    try:
        s = GraphStore()
        s.ensure_schema()
        return s
    except Exception as e:
        st.error(f"Neo4j 연결 실패: {e}")
        st.info("오프라인 모드로 계속합니다(그래프 기능 비활성).")
        class NullStore:
            def ensure_schema(self): pass
            def upsert_drug(self, *a, **k): return {}
            def log_query_and_result(self, *a, **k): return ""
            def find_interactions_for_drug(self, *a, **k): return []
            def get_chunks_for_drug(self, *a, **k): return []
            def get_drug_node(self, *a, **k): return {}
            def get_user_history(self, *a, **k): return []
            def resolve_drug_name(self, *a, **k): return None
            def upsert_verification(self, *a, **k): return None
        return NullStore()

store = get_store()

# per-session user id
if "user_id" not in st.session_state:
    st.session_state["user_id"] = "u_" + uuid.uuid4().hex[:8]

# LangGraph workflow
graph = build_graph()

st.set_page_config(page_title="약물 상호작용 분석기", layout="wide")
st.title("💊 약물 상호작용 분석기")

tabs = st.tabs(["🧪 질의", "📥 인덱스(텍스트)"])

with tabs[1]:
    st.markdown("#### 텍스트를 입력하거나 .txt 파일을 업로드하여 그래프에 인덱싱합니다.")
    colA, colB = st.columns([2,1])
    with colA:
        text_input = st.text_area("원문 텍스트", height=220, placeholder="의약 품목설명서/논문 일부를 붙여넣기...")
    with colB:
        up = st.file_uploader("또는 .txt 업로드", type=["txt"])
        if up is not None and not text_input:
            text_input = up.read().decode("utf-8", errors="ignore")
    doc_id = st.text_input("문서 ID", value=str(uuid.uuid4())[:8])
    title = st.text_input("제목(옵션)", value="사용자 문서")
    src_url = st.text_input("출처 URL(옵션)", value="")
    if st.button("📥 인덱싱 실행"):
        if not text_input.strip():
            st.warning("텍스트가 비어 있습니다.")
        else:
            with st.spinner("LLM 추출 → 그래프 적재 중..."):
                CHUNK = 1400
                all_text = text_input.strip()
                chunks = [all_text[i:i+CHUNK] for i in range(0, len(all_text), CHUNK)]
                results = []
                for ci, ch in enumerate(chunks):
                    state = {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}:{ci}",
                        "text": ch,
                        "title": title,
                        "source_url": src_url
                    }
                    out = index_text_chunk(state)
                    results.append(out)
                st.success(f"총 {len(chunks)}개 청크 인덱싱 완료. 추출 개요: {results}")

st.markdown("<style>hr{margin-top:.9rem;margin-bottom:.9rem;opacity:.6}</style>", unsafe_allow_html=True)

st.markdown("""
#### 💬 어떤 약물(약품)에 대해 궁금하세요? 또는 두 약물의 상호관계를 알고 싶으신가요?
- **한 가지 약물**만 궁금하시면 👉 왼쪽 입력칸에만 입력해 주세요.  
- **약물 간 상호작용**이 궁금하면 👉 오른쪽 입력칸도 함께 입력해 주세요.
""")

col1, col2 = st.columns(2)
with col1:
    drug1 = st.text_input("🩺 약물(약품) 1", placeholder="예: 타이레놀")
with col2:
    drug2 = st.text_input("🩺 약물(약품) 2", placeholder="예: 이부프로펜")

with tabs[0]:
    if st.button("🔍 분석하기"):
        if not drug1:
            st.warning("⚠️ 약물 1은 반드시 입력해야 합니다.")
            st.stop()

        with st.spinner("💬 답변을 생성 중입니다..."):
            try:
                inputs = {"drug1": drug1}
                if drug2:
                    inputs["drug2"] = drug2

                result = graph.invoke(inputs)["result"]
                result = result.replace("```", "")

                if drug2 and "### 📌 약물 1:" in result and "### 📌 약물 2:" in result:
                    try:
                        parts = result.split("### 📌 약물 1:")[1].split("### 📌 약물 2:")
                        drug1_info = parts[0].strip()
                        rest = parts[1].split("### 💥 두 약물의 상호작용")
                        drug2_info = rest[0].strip()
                        interaction_info = "### 💥 두 약물의 상호작용" + rest[1].strip()

                        interaction_info = re.sub(
                            r"^(###\s*💥\s*두 약물의 상호작용)\s*[-–—:]*\s*",
                            r"\1\n\n",
                            interaction_info,
                            flags=re.MULTILINE,
                        )

                        col1a, col2a = st.columns([1, 1])
                        with col1a:
                            st.markdown(f"### {drug1}\\n\\n{drug1_info}", unsafe_allow_html=True)
                        with col2a:
                            st.markdown(f"### {drug2}\\n\\n{drug2_info}", unsafe_allow_html=True)

                        st.markdown("---")
                        col1b, col2b = st.columns([1, 1])
                        with col1b:
                            render_db_info(drug1)
                        with col2b:
                            render_db_info(drug2)

                        st.markdown("---")
                        st.markdown(interaction_info, unsafe_allow_html=True)

                        # log (two-drug)
                        try:
                            hit1 = fuzzy_find(drug1, topn=1)
                            hit2 = fuzzy_find(drug2, topn=1)
                            row1 = _first_row(hit1)
                            row2 = _first_row(hit2)

                            # 주성분(가능한 컬럼 후보들을 넉넉히 커버)
                            ing1 = _pick(row1, "INGREDIENT", "ingredient", "주성분", "성분")
                            ing2 = _pick(row2, "INGREDIENT", "ingredient", "주성분", "성분")

                            # 주의/첨부 문서 URL(프로바이더마다 컬럼명이 다를 수 있음)
                            url1 = _pick(row1, "WARN_URL", "warn_url", "주의사항URL", "허가사항URL", "PDF_URL", "첨부문서URL", "첨부문서")
                            url2 = _pick(row2, "WARN_URL", "warn_url", "주의사항URL", "허가사항URL", "PDF_URL", "첨부문서URL", "첨부문서")


                            sections = {
                                "drug1_card": f"### {drug1}\\n\\n{drug1_info}",
                                "drug2_card": f"### {drug2}\\n\\n{drug2_info}",
                                "interaction_md": interaction_info,
                                "ingredient1": ing1, "ingredient2": ing2,
                                "warn_url1": url1, "warn_url2": url2,
                            }

                            store.upsert_drug(drug1, ing1)
                            store.upsert_drug(drug2, ing2)
                            store.log_query_and_result(
                                user_id=st.session_state["user_id"],
                                text=f"{drug1} vs {drug2}",
                                drug1_display=drug1,
                                drug2_display=drug2,
                                sections=sections,
                            )
                        except Exception as e:
                            st.caption(f"⚠️ 그래프 저장(2-약물) 실패: {e}")

                    except Exception:
                        st.warning("⚠️ 응답 파싱 중 문제가 발생했습니다. 전체 내용을 출력합니다.")
                        st.markdown(result)

                else:
                    # single-drug mode
                    col1a, _ = st.columns([1, 1])
                    with col1a:
                        st.markdown(result, unsafe_allow_html=True)
                    st.markdown("---")
                    col1b, _ = st.columns([1, 1])
                    with col1b:
                        render_db_info(drug1)

                    # log (single-drug)
                    try:
                        hit1 = fuzzy_find(drug1, topn=1)
                        row1 = _first_row(hit1)
                        ing1 = _pick(row1, "INGREDIENT", "ingredient", "주성분", "성분")
                        url1 = _pick(row1, "WARN_URL", "warn_url", "주의사항URL", "허가사항URL", "PDF_URL", "첨부문서URL", "첨부문서")


                        sections = {
                            "drug1_card": result,
                            "ingredient1": ing1,
                            "warn_url1": url1,
                        }

                        store.upsert_drug(drug1, ing1)
                        store.log_query_and_result(
                            user_id=st.session_state["user_id"],
                            text=drug1,
                            drug1_display=drug1,
                            drug2_display=None,
                            sections=sections,
                        )
                    except Exception as e:
                        st.caption(f"⚠️ 그래프 저장(단일 약물) 실패: {e}")

            except Exception as e:
                st.error(f"❗ 오류 발생: {e}")
    else:
        st.info("ℹ️ 위에 약물명을 입력하고 '🔍 분석하기' 버튼을 눌러주세요.")