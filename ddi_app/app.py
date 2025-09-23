# app.py
import asyncio
from uuid import uuid4
import streamlit as st
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from ddi_app.pipeline import build_workflow, AppState

st.set_page_config(page_title="약물 상호작용 분석기", layout="wide")
st.title("💊 약물 상호작용 분석기")
st.markdown('''
#### 💬 어떤 약물(약품)에 대해 궁금하세요? 또는 두 약물의 상호관계를 알고 싶으신가요?
- **한 가지 약물**만 궁금하시면 👉 왼쪽 입력칸에만 입력해 주세요.  
- **약물 간 상호작용**이 궁금하면 👉 오른쪽 입력칸도 함께 입력해 주세요.
''')

col1, col2 = st.columns(2)
with col1:
    drug1 = st.text_input("🩺 약물(약품) 1", placeholder="예: 타이레놀").strip()
with col2:
    drug2 = st.text_input("🩺 약물(약품) 2", placeholder="예: 이부프로펜").strip()
btn = st.button("🔍 분석하기")

# Session-scoped IDs
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"ui-{uuid4()}"
st.session_state.setdefault("checkpoint_ns", "streamlit-ddi")

async def run_pipeline_once(init: AppState, thread_id: str, checkpoint_ns: str):
    # Create saver and use it in the SAME loop as ainvoke
    async with AsyncSqliteSaver.from_conn_string("langgraph_checkpoints.db") as saver:
        await saver.setup()  # eagerly start aiosqlite thread once, in this loop
        workflow = build_workflow()
        app = workflow.compile(checkpointer=saver)
        result = await app.ainvoke(
            init,
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                }
            },
        )
        return result

if btn:
    if not drug1:
        st.warning("⚠️ 약물 1은 반드시 입력해야 합니다.")
        st.stop()
    with st.spinner("💬 분석 파이프라인 실행 중..."):
        init = AppState(drug1=drug1, drug2=(drug2 or None))
        final_state = asyncio.run(
            run_pipeline_once(
                init,
                st.session_state["thread_id"],
                st.session_state["checkpoint_ns"],
            )
        )
        state = final_state if isinstance(final_state, dict) else final_state.dict()

    st.markdown("---")
    if drug2:
        try:
            answer_md = state.get("answer_md", "") or ""
            parts = answer_md.split(f"### 📌 약물 2: {drug2}")
            left = parts[0]
            rest = f"### 📌 약물 2: {drug2}" + (parts[1] if len(parts) > 1 else "")
            right, *tail = rest.split("### 💥 두 약물의 상호작용")
            with col1:
                st.markdown(left, unsafe_allow_html=True)
            with col2:
                st.markdown(right, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### 💥 두 약물의 상호작용\n" + (tail[0] if tail else ""), unsafe_allow_html=True)
        except Exception:
            st.markdown(answer_md, unsafe_allow_html=True)
    else:
        with col1:
            st.markdown(state.get("answer_md", ""), unsafe_allow_html=True)


