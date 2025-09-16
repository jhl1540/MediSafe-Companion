import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- API 키 로드 ---
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true" # LangSmith 추적을 위해 추가 (선택 사항)

# --- 1. Graph State 정의 ---
# 대화의 흐름을 메시지 리스트로 관리합니다.
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], lambda x, y: x + y]

# --- 2. 도구(Tools) 및 LLM 준비 ---
# 사용할 도구를 정의합니다. Tavily 검색 도구를 사용합니다.
tool = TavilySearchResults(max_results=2)
tools = [tool]

# 도구를 사용할 수 있는 LLM을 준비합니다.
llm = ChatOpenAI(model="gpt-4o")
# .bind_tools는 LLM이 도구의 스키마를 보고 언제 호출할지 판단하도록 만듭니다.
llm_with_tools = llm.bind_tools(tools)


# --- 3. LangGraph 노드 정의 ---

# (1) Chatbot 노드: LLM을 호출하여 응답을 생성합니다.
def chatbot(state: AgentState):
    """LLM을 호출하여 사용자의 다음 행동을 결정하고 메시지를 생성합니다."""
    print("-> 노드: chatbot")
    # 현재까지의 메시지를 LLM에 전달
    response = llm_with_tools.invoke(state['messages'])
    # LLM의 응답을 메시지 리스트에 추가
    return {"messages": [response]}

# (2) Tools 노드: Chatbot이 도구 사용을 결정했을 때 실제 도구를 실행합니다.
# LangGraph에서 제공하는 ToolNode를 사용하면 편리합니다.
tool_node = ToolNode(tools)

# (3) 조건부 엣지(Router) 함수: 도구를 사용할지, 아니면 끝낼지 결정합니다.
def should_continue(state: AgentState):
    """LLM의 마지막 응답을 보고 다음 경로를 결정합니다."""
    print("-> 조건부 엣지: should_continue")
    last_message = state['messages'][-1]
    # 마지막 메시지에 tool_calls가 있다면, 도구를 실행해야 합니다.
    if last_message.tool_calls:
        print("-> 경로: tools 노드로 이동")
        return "tools"
    # tool_calls가 없다면, 사용자에게 답변한 것이므로 종료합니다.
    else:
        print("-> 경로: END로 이동")
        return END

# --- 4. Graph 구성 ---
workflow = StateGraph(AgentState)

# 노드를 추가합니다.
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", tool_node)

# 그래프의 시작점을 'chatbot' 노드로 설정합니다.
workflow.set_entry_point("chatbot")

# 조건부 엣지를 추가합니다.
workflow.add_conditional_edges(
    "chatbot",         # 'chatbot' 노드 다음에
    should_continue,   # 'should_continue' 함수를 실행하여 경로를 결정
)

# 'tools' 노드는 항상 'chatbot' 노드로 다시 돌아갑니다. (Agentic Loop)
workflow.add_edge("tools", "chatbot")

# 그래프를 컴파일합니다.
app = workflow.compile()


# --- 5. 직접 실행할 때만 작동하도록 설정 ---
if __name__ == "__main__":
    print("--- LangGraph Agent 실행 (직접 모드) ---")
    
    # HumanMessage 객체를 사용하여 실행해야 합니다.
    inputs = {"messages": [HumanMessage(content="서울의 현재 날씨는 어때?")]}
    
    # stream을 통해 각 단계의 출력을 실시간으로 확인
    for output in app.stream(inputs):
        # output은 모든 노드의 출력을 포함합니다. 마지막 키의 값을 출력합니다.
        for key, value in output.items():
            print(f"출력 노드: '{key}'")
            print("---")
            print(value)
    print("\n--- 실행 종료 ---")