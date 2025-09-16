import os
import json
from dotenv import load_dotenv
from typing import TypedDict, List

# --- 시각화를 위해 추가 ---
from IPython.display import Image, display

from langchain_core.documents import Document
from langchain.tools import tool
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
import uvicorn
from fastapi import FastAPI
from langserve import add_routes

# --- API 키 및 LangSmith 환경 변수 로드 ---
load_dotenv()

# --- 1. Graph State 정의 ---
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str

# --- 2. 도구(Tools) 및 Retriever 준비 ---
# ... (기존 코드와 동일)
loader = CSVLoader(file_path='db_drug_interactions.csv', encoding='utf-8')
docs = loader.load()[:1000]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={'k': 5})

@tool
def local_db_search(query: str) -> List[Document]:
    """Searches the local drug interaction database."""
    print(f"-> 로컬 DB 검색 실행: {query}")
    return retriever.invoke(query)

@tool
def web_search(query: str) -> List[Document]:
    """Searches the web for the latest medical information."""
    print(f"-> 웹 검색 실행: {query}")
    tavily_tool = TavilySearchResults(max_results=3)
    results = tavily_tool.invoke({"query": query})
    return [Document(page_content=d["content"], metadata={"source": d["url"]}) for d in results]

# --- 3. LLM 준비 ---
llm = ChatOpenAI(model="gpt-4o")

# --- 4. LangGraph 노드 정의 ---
# ... (기존 코드와 동일)
def route_query(state: AgentState) -> str:
    print("-> 노드: route_query")
    prompt = f"""사용자의 다음 질문을 분석하여 어떤 도구를 사용해야 할지 결정하세요.
    - 'local_db_search': 두 가지 이상의 특정 약물 이름 간의 상호작용에 대한 질문일 경우.
    - 'web_search': 최신 뉴스, 일반적인 정보, 단일 약물에 대한 질문일 경우.
    질문: {state['question']}
    선택 (local_db_search 또는 web_search 만 반환):"""
    response = llm.invoke(prompt)
    decision = response.content.strip()
    print(f"-> 라우팅 결정: {decision}")
    if "local_db_search" in decision:
        return "local_db_search"
    else:
        return "web_search"

def local_db_node(state: AgentState) -> dict:
    print("-> 노드: local_db_search")
    documents = local_db_search.invoke(state['question'])
    return {"documents": documents}

def web_search_node(state: AgentState) -> dict:
    print("-> 노드: web_search")
    documents = web_search.invoke(state['question'])
    return {"documents": documents}

def synthesize_response(state: AgentState) -> dict:
    print("-> 노드: synthesize_response")
    context = "\n\n".join([doc.page_content for doc in state['documents']])
    prompt = f"""주어진 정보만을 바탕으로 다음 질문에 대해 답변해 주세요.
    [정보]: {context}
    [질문]: {state['question']}"""
    response = llm.invoke(prompt)
    return {"generation": response.content}

# --- 5. Graph 구성 ---
workflow = StateGraph(AgentState)
workflow.add_node("local_db_search", local_db_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("synthesize_response", synthesize_response)
workflow.set_conditional_entry_point(
    route_query,
    {"local_db_search": "local_db_search", "web_search": "web_search"}
)
workflow.add_edge("local_db_search", "synthesize_response")
workflow.add_edge("web_search", "synthesize_response")
workflow.add_edge("synthesize_response", END)
app = workflow.compile()


# --- ⭐️ 그래프 시각화 코드 추가 ⭐️ ---
try:
    # 그래프 구조를 PNG 이미지 파일로 저장합니다.
    graph_image_bytes = app.get_graph().draw_png()
    with open("medical_agent_graph.png", "wb") as f:
        f.write(graph_image_bytes)
    print("✅ 그래프 이미지가 'medical_agent_graph.png' 파일로 저장되었습니다.")
    # (선택사항) Jupyter Notebook 환경이라면 아래 코드로 바로 이미지를 표시할 수 있습니다.
    # display(Image(graph_image_bytes))
except ImportError as e:
    print(f"⚠️ 그래프 시각화에 실패했습니다. 'pygraphviz' 라이브러리가 필요합니다. (에러: {e})")
except Exception as e:
    print(f"⚠️ 그래프를 그리는 중 에러가 발생했습니다: {e}")


# --- 6. FastAPI 서버 설정 ---
api = FastAPI(
  title="Medical Agent Server",
  version="1.0",
  description="A server for the medical information agent.",
)
add_routes(
    api,
    app,
    path="/agent",
)

# --- 7. 서버 실행 ---
if __name__ == "__main__":
    print("--- FastAPI 서버를 시작합니다 ---")
    print("Playground UI: http://1227.0.0.1:8000/agent/playground/")
    uvicorn.run(api, host="0.0.0.0", port=8000)