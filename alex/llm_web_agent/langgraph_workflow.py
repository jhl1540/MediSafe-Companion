import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langgraph.graph import StateGraph, END
from prompt_templates import single_drug_prompt, interaction_prompt

# 🔐 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 💬 LLM 인스턴스
llm = ChatOpenAI(
    openai_api_key=api_key,
    model="gpt-4o",
    temperature=0.5
)

# 🧠 약사 역할 설정
system_msg = SystemMessage(content="너는 약물 정보와 상호작용을 전문가처럼 설명하는 약사야.")

# 🌐 상태머신 노드 정의
def analyze_single_drug(state):
    drug = state["drug1"]
    prompt = single_drug_prompt.format(drug=drug)
    response = llm.predict_messages([system_msg], prompt)
    return {"result": response.content}

def analyze_two_drugs(state):
    drug1 = state["drug1"]
    drug2 = state["drug2"]
    prompt = interaction_prompt.format(drug1=drug1, drug2=drug2)
    response = llm.predict_messages([system_msg], prompt)
    return {"result": response.content}

# 🔄 LangGraph 그래프 구축
def build_graph():
    workflow = StateGraph()

    workflow.add_node("analyze_single", analyze_single_drug)
    workflow.add_node("analyze_interaction", analyze_two_drugs)

    # 조건 분기
    def should_use_interaction(state):
        return "drug2" in state and state["drug2"] not in ("", None)

    workflow.add_conditional_edges(
        "entry", should_use_interaction,
        {
            True: "analyze_interaction",
            False: "analyze_single"
        }
    )

    # 종료 처리
    workflow.set_entry_point("entry")
    workflow.add_edge("analyze_single", END)
    workflow.add_edge("analyze_interaction", END)

    return workflow.compile()