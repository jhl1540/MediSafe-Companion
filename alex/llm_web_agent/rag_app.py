import os
import uuid
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from neo4j_store import GraphStore
from db_utils import fuzzy_find
from tavily import TavilyClient

from prompt_templates import (
    graphqa_router_system, graphqa_router_user,
    sidefx_system, sidefx_user,
    patient_impact_system, patient_impact_user,
    web_verify_system, web_verify_user
)

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(page_title="GraphRAG 약물 지식 질의", layout="wide")
st.title("📚 GraphRAG 약물 지식 질의")

# Per-session user id (used for prescription/query history)
if "user_id" not in st.session_state:
    st.session_state["user_id"] = "u_" + uuid.uuid4().hex[:8]

@st.cache_resource
def get_store() -> GraphStore:
    s = GraphStore()
    s.ensure_schema()
    return s

store = get_store()

@st.cache_resource
def get_llm():
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

llm = get_llm()

@st.cache_resource
def get_tavily():
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return None
    return TavilyClient(api_key=key)

tavily = get_tavily()

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("자유롭게 질문해 보세요 (예: What are the side effects of metformin? / What medications interact with warfarin? / 처방 내역).")
q = st.text_input("🧠 질문 (영/한 자유)", placeholder="예: What medications interact with warfarin?")
go = st.button("질의 실행")

st.write("")
web_verify = st.checkbox("🌐 웹 검증/보강 실행", value=False, help="그래프 결과를 웹 문헌으로 교차 검증하고 그래프에 기록합니다.")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _route(question: str) -> dict:
    msgs = [SystemMessage(content=graphqa_router_system),
            HumanMessage(content=graphqa_router_user.format(question=question))]
    txt = llm.invoke(msgs).content.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.startswith("json"):
            txt = txt[4:]
    import json
    try:
        return json.loads(txt)
    except Exception:
        return {"tool": "interactions", "args": {"drug": question}}

def _canon(name: str) -> str:
    hit = store.resolve_drug_name(name)
    if hit:
        return hit.get("display_name") or hit.get("name") or name
    aliases = {
        "warfarin": "와파린",
        "metformin": "메트포르민",
        "ibuprofen": "이부프로펜",
        "aspirin": "아스피린",
        "acetaminophen": "아세트아미노펜",
        "paracetamol": "아세트아미노펜",
        "ethanol": "에탄올",
        "nicotine": "니코틴",
    }
    k = (name or "").strip().lower()
    return aliases.get(k, name)

def _gather_interactions(drug: str):
    return store.find_interactions_for_drug(drug)

def _gather_chunks(drug: str, k: int = 8):
    return store.get_chunks_for_drug(drug, k=k)

def _format_evidence(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        snippet = (c.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 380:
            snippet = snippet[:380] + "..."
        lines.append(f"- [{c.get('chunk_id','?')}] {snippet}  (src: {c.get('source_url','')})")
    return "\n".join(lines)

def _strip_first_header(md: str) -> str:
    if not md:
        return ""
    lines = md.lstrip().splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        return "\n".join(lines[1:]).lstrip()
    return md

def _tavily_search(q: str, include_domains=None, max_results=5) -> list[dict]:
    if tavily is None:
        return []
    params = {}
    if include_domains:
        params["include_domains"] = include_domains
    res = tavily.search(q, search_depth="advanced", max_results=max_results, **params)
    return res.get("results", []) if isinstance(res, dict) else []

def _format_snippets_for_llm(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        title = h.get("title","").strip()
        url   = h.get("url","").strip()
        content = (h.get("content","") or "").strip().replace("\n"," ")
        if len(content) > 380:
            content = content[:380] + "..."
        lines.append(f"- {title}\n  URL: {url}\n  SNIPPET: {content}")
    return "\n".join(lines)

def _web_verify_pair(a: str, b: str, graph_md: str) -> dict:
    hits = _tavily_search(
        f"{a} {b} drug interaction OR coadministration site:drugs.com OR site:dailymed.nlm.nih.gov OR site:pubmed.ncbi.nlm.nih.gov OR site:fda.gov",
        max_results=6
    )
    if not hits:
        hits = _tavily_search(f'{a} {b} drug interaction', max_results=6)

    if not hits:
        return {"status":"insufficient","summary":"웹 검색 결과가 부족합니다.","citations":[]}

    snippets = _format_snippets_for_llm(hits)
    msgs = [
        SystemMessage(content=web_verify_system),
        HumanMessage(content=web_verify_user.format(a=a, b=b, graph_md=graph_md or "(none)", snippets=snippets))
    ]
    raw = llm.invoke(msgs).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    import json
    try:
        data = json.loads(raw)
    except Exception:
        data = {"status":"insufficient","summary":"검증 파서 오류","citations":[]}
    cits = data.get("citations") or []
    if isinstance(cits, list):
        cits = cits[:5]
    data["citations"] = cits
    return data

def verify_and_update_from_web(drug: str) -> list[dict]:
    rows = _gather_interactions(drug)
    reports = []
    for r in rows:
        a = drug
        b = r["interacts_with"]
        graph_md = r.get("interaction_md","")
        verdict = _web_verify_pair(a, b, graph_md)
        try:
            store.upsert_verification(a, b, verdict.get("status","insufficient"),
                                      verdict.get("summary",""), verdict.get("citations",[]))
        except Exception:
            pass
        reports.append({"other": b, **verdict})
    return reports

# ─────────────────────────────────────────────────────────────────────────────
# Answer builders
# ─────────────────────────────────────────────────────────────────────────────
def answer_side_effects(drug: str):
    ev = _gather_chunks(drug)
    if ev:
        msgs = [SystemMessage(content=sidefx_system),
                HumanMessage(content=sidefx_user.format(drug=drug, evidence=_format_evidence(ev)))]
        return llm.invoke(msgs).content

    node = store.get_drug_node(drug)
    if node and node.get("card"):
        return f"### {node.get('display_name', drug)} — 부작용(요약)\n\n{_strip_first_header(node['card'])}"
    return f"그래프에 '{drug}' 관련 텍스트/카드가 아직 없습니다. 인덱싱 탭에서 문서 또는 질의를 추가해 주세요."

def answer_interactions(drug: str):
    rows = _gather_interactions(drug)
    if not rows:
        return (
            f"그래프에 '{drug}'의 상호작용 기록이 없습니다. "
            "두 약물 질의를 통해 기록을 쌓거나, 인덱싱 후 추출 파이프라인을 사용해 보세요."
        )
    out = [f"**{rows[0]['drug']}**의 상호작용:"]
    for r in rows:
        line = f"- ↔ **{r['interacts_with']}**"
        sev = (r.get("severity") or "").strip()
        if sev and sev.lower() not in ("unknown",):
            line += f" · 중증도: **{sev}**"
        imd = (r.get("interaction_md") or "").strip()
        if imd:
            body = _strip_first_header(imd).splitlines()
            first = body[0].strip() if body else ""
            if first:
                if len(first) > 240:
                    first = first[:240] + "…"
                line += f"\n    - 요약: {first}"
        out.append(line)
    return "\n".join(out)

def answer_patient_impact(question: str, drug: str, age: int | None, sex: str | None):
    ev = _gather_chunks(drug)
    rows = _gather_interactions(drug)
    i_md = "\n".join([_strip_first_header(r.get("interaction_md","")) for r in rows if r.get("interaction_md")])[:1600]
    msgs = [SystemMessage(content=patient_impact_system),
            HumanMessage(content=patient_impact_user.format(
                question=question, drug=drug, age=age or "unknown", sex=sex or "unknown",
                interaction_md=i_md, evidence=_format_evidence(ev)
            ))]
    return llm.invoke(msgs).content

def answer_prescription_history(user_id: str):
    rows = store.get_user_history(user_id, limit=30)
    if not rows:
        return "아직 기록된 처방/질의 내역이 없습니다."
    out = []
    for r in rows:
        ts = str(r.get("ts") or "")[:19].replace("T", " ")
        mode = r.get("mode", "single")
        if mode == "pair" and (r.get("drug1") and r.get("drug2")):
            imd = _strip_first_header((r.get("interaction_md") or "").strip())
            first_line = imd.splitlines()[0] if imd else ""
            snippet = (first_line[:160] + "…") if len(first_line) > 160 else first_line
            out.append(f"- [{ts}] **{r['drug1']} ↔ {r['drug2']}**  • {snippet or '상세 요약 없음'}")
        else:
            out.append(f"- [{ts}] **{r.get('drug1','(약물 미상)')}**")
    return "\n".join(out)

# ─────────────────────────────────────────────────────────────────────────────
# Simple heuristics
# ─────────────────────────────────────────────────────────────────────────────
HISTORY_KWS = ["처방 내역", "처방내역", "내 처방", "내역 보기", "history", "my prescriptions", "my meds", "기록"]
def _looks_like_history(text: str) -> bool:
    t = text or ""
    tl = t.lower()
    return any(kw in t or kw in tl for kw in HISTORY_KWS)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if go and q.strip():
    with st.spinner("그래프에서 답변 구성 중..."):
        if _looks_like_history(q):
            route = {"tool": "prescription_history", "args": {}}
        else:
            route = _route(q)

        tool = route.get("tool", "interactions")
        args = route.get("args", {})

        try:
            if tool == "side_effects":
                drug = _canon(args.get("drug", "")) or _canon(q)
                st.markdown(f"**[side_effects]** 대상: {drug}")
                st.markdown(answer_side_effects(drug), unsafe_allow_html=True)

            elif tool == "patient_impact":
                drug = _canon(args.get("drug", "")) or _canon(q)
                age = args.get("age")
                sex = args.get("sex")
                st.markdown(f"**[patient_impact]** 대상: {drug} · age={age} · sex={sex}")
                st.markdown(answer_patient_impact(q, drug, age, sex), unsafe_allow_html=True)

            elif tool == "prescription_history":
                st.markdown("**[prescription_history]** 현재 세션의 처방/질의 내역")
                st.markdown(answer_prescription_history(st.session_state["user_id"]), unsafe_allow_html=True)

            else:  # interactions (기본)
                drug = _canon(args.get("drug", "")) or _canon(q)
                st.markdown(f"**[interactions]** 대상: {drug}")
                st.markdown(answer_interactions(drug), unsafe_allow_html=True)

                # (옵션) 웹 검증 실행
                if web_verify:
                    with st.spinner("🌐 웹 문헌으로 교차 검증 중..."):
                        reports = verify_and_update_from_web(drug)
                    st.markdown("#### 🌐 웹 검증 결과")
                    for rep in reports:
                        status = rep.get("status","insufficient")
                        badge = {"support":"✅ 지원", "contradict":"❌ 상충", "insufficient":"⚪ 보완 필요"}.get(status,"⚪ 보완 필요")
                        with st.expander(f"{badge}  {drug} ↔ {rep['other']}", expanded=False):
                            st.write(rep.get("summary","(요약 없음)"))
                            cits = rep.get("citations") or []
                            if cits:
                                st.caption("참고 링크:")
                                for u in cits:
                                    st.markdown(f"- {u}")

        except Exception as e:
            st.error(f"오류: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Graph browse (helper view)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔎 그래프 직접 탐색 (보조 보기)")

drug_lookup = st.text_input("약물명(그래프 조회용)", value="")
if st.button("그래프 조회"):
    if not drug_lookup.strip():
        st.warning("약물명을 입력하세요.")
    else:
        rows = store.find_interactions_for_drug(drug_lookup)
        if rows:
            for r in rows:
                pair_title = f"**{r['drug']} ↔ {r['interacts_with']}**"
                sev = (r.get("severity") or "").strip()
                imd = (r.get("interaction_md") or "").strip()

                with st.expander(pair_title, expanded=False):
                    if imd:
                        st.markdown(_strip_first_header(imd), unsafe_allow_html=True)
                    else:
                        if sev and sev.lower() != "unknown":
                            st.write(f"심각도: **{sev}**")
                        mech = (r.get("mechanism") or "").strip()
                        mgmt = (r.get("management") or "").strip()
                        if mech or mgmt:
                            st.caption(f"기전: {mech}  |  관리: {mgmt}  |  출처: {r.get('source','')}")
                        if not (sev and sev.lower() != "unknown") and not (mech or mgmt):
                            st.info("이 상호작용에 대한 상세 요약(interaction_md)이 아직 없습니다.")
        else:
            st.info("현재 그래프DB에 등록된 상호작용이 없습니다.")