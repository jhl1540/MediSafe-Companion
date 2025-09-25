import os
from dotenv import load_dotenv

# 최신 경로/타입
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# LangGraph 시작점 분기(START) 사용
from langgraph.graph import StateGraph, START, END

from prompt_templates import single_drug_prompt, interaction_prompt

# 🔐 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 💬 LLM 인스턴스
llm = ChatOpenAI(
    api_key=api_key,   # openai_api_key=... 대신 최신 키워드
    model="gpt-4o",
    temperature=0.5
)

# 🧠 약사 역할
system_msg = SystemMessage(content="너는 약물 정보와 상호작용을 전문가처럼 설명하는 약사야.")

# 🌐 상태머신 노드
def analyze_single_drug(state: dict):
    drug = state["drug1"]
    prompt_text = single_drug_prompt.format(drug=drug)
    messages = [system_msg, HumanMessage(content=prompt_text)]
    response = llm.invoke(messages)
    return {"result": response.content}

def analyze_two_drugs(state: dict):
    drug1 = state["drug1"]
    drug2 = state["drug2"]
    prompt_text = interaction_prompt.format(drug1=drug1, drug2=drug2)
    messages = [system_msg, HumanMessage(content=prompt_text)]
    response = llm.invoke(messages)
    return {"result": response.content}

# 🔄 그래프 구축
def build_graph():
    workflow = StateGraph(dict)

    workflow.add_node("analyze_single", analyze_single_drug)
    workflow.add_node("analyze_interaction", analyze_two_drugs)

    def should_use_interaction(state: dict):
        return "drug2" in state and state["drug2"] not in ("", None)

    workflow.add_conditional_edges(
        START,
        should_use_interaction,
        {
            True: "analyze_interaction",
            False: "analyze_single"
        }
    )

    workflow.add_edge("analyze_single", END)
    workflow.add_edge("analyze_interaction", END)

    return workflow.compile()
