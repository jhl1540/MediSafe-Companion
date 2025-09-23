# DDI LangGraph + CSV Cache + Neo4j

## Quickstart
```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env with your keys
langgraph dev
# -> API: http://localhost:2024
# -> Docs: http://localhost:2024/docs
# -> Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

## Scripts
Normalize DB.csv:
```bash
python scripts/normalize_db_csv.py --csv /mnt/data/DB.csv
```

Ingest health.kr CSV to Neo4j:
```bash
python scripts/ingest_healthkr_to_neo4j.py --csv /path/to/healthkr.csv \
  --neo4j-uri neo4j://localhost:7687 --neo4j-user neo4j --neo4j-pass password
```
