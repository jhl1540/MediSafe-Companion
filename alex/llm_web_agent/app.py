import streamlit as st
from langgraph_workflow import build_graph

# LangGraph 워크플로우 생성
graph = build_graph()

st.set_page_config(page_title="약물 상호작용 분석기", layout="wide")
st.title("💊 약물 상호작용 분석기")

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

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown(f"### {drug1}\n\n{drug1_info}", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"### {drug2}\n\n{drug2_info}", unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown(interaction_info, unsafe_allow_html=True)
                except:
                    st.warning("⚠️ 응답 파싱 중 문제가 발생했습니다. 전체 내용을 출력합니다.")
                    st.markdown(result)
            else:
                col1, _ = st.columns([1, 1])
                with col1:
                    st.markdown(result, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❗ 오류 발생: {e}")
else:
    st.info("ℹ️ 위에 약물명을 입력하고 '🔍 분석하기' 버튼을 눌러주세요.")