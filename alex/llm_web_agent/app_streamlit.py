import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

# 📌 환경 변수 로딩
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 🔑 OpenAI client 생성
client = OpenAI(api_key=api_key)

# 🎨 Streamlit 기본 설정
st.set_page_config(page_title="약물 상호작용 분석기", layout="wide")
st.title("💊 약물 상호작용 분석기")

st.markdown("""
#### 💬 어떤 약물(약품)에 대해 궁금하세요? 또는 두 약물의 상호관계를 알고 싶으신가요?
- **한 가지 약물**만 궁금하시면 👉 왼쪽 입력칸에만 입력해 주세요.  
- **약물 간 상호작용**이 궁금하면 👉 오른쪽 입력칸도 함께 입력해 주세요.
""")

# 📥 입력창
col1, col2 = st.columns(2)
with col1:
    drug1 = st.text_input("🩺 약물(약품) 1", placeholder="예: 타이레놀")
with col2:
    drug2 = st.text_input("🩺 약물(약품) 2", placeholder="예: 이부프로펜")

# 🔍 버튼
generate = st.button("🔍 분석하기")

if generate:
    if not drug1:
        st.warning("⚠️ 약물 1은 반드시 입력해야 합니다.")
        st.stop()

    # 🧠 프롬프트 구성
    if drug1 and not drug2:
        prompt = f"""
'{drug1}'이라는 약물에 대해 아래 정보를 항목별로 알려줘:

1. 💊 **주요 약품명** (예시 2개)  
2. 😷 **복용 증상 또는 상황**  
3. 💡 **효과/효능**  
4. ⚠️ **특이사항** (처방 필요 여부, 주의사항, 피해야 할 음식 등)  

Markdown으로 정리하고, 마지막엔 📚출처도 반드시 알려줘.
"""
    else:
        prompt = f"""
'{drug1}'와 '{drug2}' 이 두 약물에 대해 아래 정보를 항목별로 정리해줘:

---

### 📌 약물 1: {drug1}

1. 💊 **주요 약품명** (예시 2개)  
2. 😷 **복용 증상 또는 상황**  
3. 💡 **효과/효능**  
4. ⚠️ **특이사항**

---

### 📌 약물 2: {drug2}

1. 💊 **주요 약품명** (예시 2개)  
2. 😷 **복용 증상 또는 상황**  
3. 💡 **효과/효능**  
4. ⚠️ **특이사항**

---

### 💥 두 약물의 상호작용  
- ✅ 함께 복용 가능 여부  
- ❌ 피해야 할 점  
- 📚 출처도 명확히 알려줘
"""

    # 🧠 OpenAI 호출
    with st.spinner("💬 답변을 생성 중입니다..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "너는 약물 정보와 상호작용을 전문가처럼 설명하는 약사야."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content
            answer = answer.replace("```", "")
            st.markdown("---")

            # ✅ 마크다운 블록을 기준으로 파싱
            if drug1 and drug2 and "### 📌 약물 1:" in answer and "### 📌 약물 2:" in answer:
                try:
                    parts = answer.split("### 📌 약물 1:")[1].split("### 📌 약물 2:")
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
                except Exception as e:
                    st.warning("⚠️ 응답 파싱 중 문제가 발생했습니다. 전체 내용을 출력합니다.")
                    st.markdown(answer)
            else:
                # 약물 1개만 입력한 경우: 왼쪽만 사용 (중앙 넘지 않도록)
                col1, _ = st.columns([1, 1])
                with col1:
                    st.markdown(answer, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❗ 오류 발생: {e}")
else:
    st.info("ℹ️ 위에 약물명을 입력하고 '🔍 분석하기' 버튼을 눌러주세요.")