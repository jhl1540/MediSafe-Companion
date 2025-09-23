# llm_websearch.py
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

def llm_search_then_summarize(query: str) -> str:
    resp = client.responses.create(
        model=MODEL,
        input=f"Search the web for reliable Korean drug monograph or interaction info about: {query}. Summarize key facts in 5 bullet points with URLs.",
        tools=[{"type":"web_search"}],   # built-in tool (supported by GPT-5 in Responses API)
    )
    return resp.output_text
