# llm_nodes.py  (Chat Completions version)

import os, json
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(), override=False)
client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # or gpt-4.1 / gpt-4o, etc.

SCHEMA = {
    "name": "DDIQueryPlan",
    "schema": {
        "type": "object",
        "properties": {
            "drugs": {"type": "array", "items": {"type": "string"}},
            "intent": {"type": "string", "enum": ["single_info","pair_ddi","unknown"]},
            "actions": {"type": "array", "items": {"type": "string"}},
            "web_search_query": {"type": "string"}  # <-- keep this
        },
        # IMPORTANT: include *all* keys that appear in properties:
        "required": ["drugs", "intent", "actions", "web_search_query"],
        "additionalProperties": False
    },
    "strict": True
}

PROMPT = """You are a clinical DDI query planner.
- Extract drug names (brand or generic), normalized as best you can (don’t over-normalize).
- intent: single_info | pair_ddi | unknown
- actions: any of healthkr:drugA, healthkr:drugB, ddinter:pair, graph:upsert
- If vague, propose web_search_query (else return an empty string). Return ONLY JSON.
"""

def llm_plan(query: str):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": query}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": SCHEMA
        }
    )
    content = resp.choices[0].message.content  # JSON string
    plan = json.loads(content)

    # defensive defaults (even though keys are required)
    plan.setdefault("drugs", [])
    plan.setdefault("actions", [])
    plan.setdefault("intent", "unknown")
    plan.setdefault("web_search_query", "")

    return plan
