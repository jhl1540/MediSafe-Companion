import os
from dotenv import load_dotenv
from typing import List, TypedDict

# --- 서버 실행을 위한 라이브러리 ---
import uvicorn
from fastapi import FastAPI
from langserve import add_routes

from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# --- 1. 환경 변수 로드 ---
load_dotenv()

# --- 2. Graph State, 도구, LLM 준비 ---
# (이전 코드와 동일)
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str

loader = CSVLoader(file_path='db_drug_interactions.csv', encoding='utf-8')
docs = loader.load()[:1000]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={'k': 5})
web_search_tool = TavilySearchResults(k=3)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# --- 3. LangGraph 노드 정의 ---
# (이전 코드와 동일)
def retrieve(state):
    print("--- 노드: retrieve ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    print("--- 노드: grade_documents ---")
    question = state["question"]
    documents = state["documents"]
    if not documents:
        print("-> 문서 없음, 웹 검색으로 라우팅")
        return "websearch"
    prompt = PromptTemplate.from_template("사용자의 질문에 대해 검색된 문서들이 관련성이 높으면 'yes', 아니면 'no'만 반환해줘.\n\n[문서]: {documents}\n[질문]: {question}")
    grader_chain = prompt | llm
    docs_str = "\n\n".join([d.page_content for d in documents])
    response = grader_chain.invoke({"documents": docs_str, "question": question})
    if "yes" in response.content.lower():
        print("-> 문서 관련성 높음, 답변 생성으로 라우팅")
        return "generate"
    else:
        print("-> 문서 관련성 낮음, 웹 검색으로 라우팅")
        return "websearch"

def generate(state):
    print("--- 노드: generate ---")
    question = state["question"]
    documents = state["documents"]
    prompt = PromptTemplate.from_template("주어진 정보만을 바탕으로 질문에 대해 답변해줘. 출처를 명시해줘.\n\n[정보]: {context}\n[질문]: {question}")
    rag_chain = prompt | llm
    docs_str = "\n\n".join([d.page_content for d in documents])
    generation = rag_chain.invoke({"context": docs_str, "question": question})
    return {"generation": generation.content}

def web_search(state):
    print("--- 노드: web_search ---")
    question = state["question"]
    web_results = web_search_tool.invoke({"query": question})
    web_docs = [Document(page_content=d["content"], metadata={"source": d["url"]}) for d in web_results]
    return {"documents": web_docs, "question": question}


# --- 4. Graph 구성 ---
# (이전 코드와 동일)
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search)
workflow.set_entry_point("retrieve")
workflow.add_conditional_edges("retrieve", grade_documents, {"generate": "generate", "websearch": "web_search"})
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()


# --- 5. FastAPI 서버 설정 ---
api = FastAPI(
  title="Self-Correcting RAG Agent Server",
  version="1.0",
  description="A server for the RAG agent with LangSmith tracing.",
)
add_routes(api, app, path="/agent")


# --- 6. 서버 실행 ---
# __name__ == "__main__" 블록 안에서 uvicorn.run을 호출하는 것이 표준 방식입니다.
if __name__ == "__main__":
    print("--- FastAPI 서버를 시작합니다 ---")
    print("Playground UI: http://127.0.0.1:8000/agent/playground/")
    uvicorn.run(api, host="0.0.0.0", port=8000)