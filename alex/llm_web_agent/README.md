# GraphMed (Recovered)

Two apps:
- `app.py`: LangGraph-based single/pair analysis + text indexing
- `rag_app.py`: GraphRAG tool router + optional web verification (Tavily)

## Quickstart
bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
cp .env.example .env  # fill keys
streamlit run app.py      # LangGraph app
streamlit run rag_app.py  # GraphRAG app