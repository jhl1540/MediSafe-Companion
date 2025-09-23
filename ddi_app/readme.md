# MediSafe Companion (Modularized)

This is a modular split of your previous single-file Streamlit app.

## Structure
- `app.py`: Streamlit entry point (UI + kicks off LangGraph)
- `ddi_app/config.py`: env + OpenAI client
- `ddi_app/db_csv.py`: CSV DB helpers
- `ddi_app/web_retrieval.py`: httpx+bs4 web fetchers
- `ddi_app/llm_backoff.py`: LLM extraction fallback
- `ddi_app/neo4j_utils.py`: Neo4j helpers
- `ddi_app/pipeline.py`: LangGraph nodes, state, graph builder
- `ddi_app/ui.py`: Markdown formatting

## Run
```bash
pip install -r requirements.txt
cp .env.example .env  # and fill values
streamlit run app.py