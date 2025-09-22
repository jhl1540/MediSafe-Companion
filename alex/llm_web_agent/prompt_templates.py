from langchain.prompts import PromptTemplate

# 단일 약물에 대한 프롬프트
single_drug_prompt = PromptTemplate(
    input_variables=["drug"],
    template="""
'{drug}'이라는 약물에 대해 아래 정보를 항목별로 알려줘:

1. 💊 **주요 약품명** (예시 2개)  
2. 😷 **복용 증상 또는 상황**  
3. 💡 **효과/효능**  
4. ⚠️ **특이사항** (처방 필요 여부, 주의사항, 피해야 할 음식 등)

Markdown으로 정리하고, 마지막엔 📚출처도 반드시 알려줘.
"""
)

# 두 약물 상호작용 프롬프트
interaction_prompt = PromptTemplate(
    input_variables=["drug1", "drug2"],
    template="""
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
)