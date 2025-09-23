\
import argparse, csv
from neo4j import GraphDatabase

SEVERITY_MAP = {
    "❌": "contraindicated",
    "금기": "contraindicated",
    "금지": "contraindicated",
    "⚠️": "major",
    "주의": "major",
    "경고": "major",
    "중등도": "moderate",
    "주의요함": "moderate",
    "경미": "minor",
    "경도": "minor",
    "없음": "none",
    "⭕": "none",
}

def normalize_severity(val: str) -> str:
    if not val:
        return ""
    t = (val or "").strip().lower()
    for k, v in SEVERITY_MAP.items():
        if k in t or k == val:
            return v
    if "금" in t:
        return "contraindicated"
    if "주의" in t or "경고" in t:
        return "major"
    return t

def read_rows(csv_path: str):
    encodings = ["utf-8-sig", "cp949", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            with open(csv_path, newline="", encoding=enc) as f:
                import csv
                reader = csv.DictReader(f)
                for r in reader:
                    yield r
            return
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--neo4j-uri", default="neo4j://localhost:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-pass", default="password")
    args = ap.parse_args()

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))

    cypher = """
    UNWIND $rows AS row
    WITH row,
         coalesce(row.`제품명2`, row.`제품명.1`) AS prod2
    MERGE (a:Drug {name: trim(row.`제품명`)})
      ON CREATE SET a.ingredient1 = coalesce(row.`성분1`,""),
                    a.ingredient2 = coalesce(row.`성분2`,""),
                    a.ingredient3 = coalesce(row.`성분3`,""),
                    a.kfda_class = coalesce(row.`식약처분류`,""),
                    a.indication1 = coalesce(row.`효능/효과1`,""),
                    a.indication2 = coalesce(row.`효능/효과2`,""),
                    a.target = coalesce(row.`대상`,"")
      ON MATCH SET  a.ingredient1 = coalesce(row.`성분1`, a.ingredient1),
                    a.ingredient2 = coalesce(row.`성분2`, a.ingredient2),
                    a.ingredient3 = coalesce(row.`성분3`, a.ingredient3),
                    a.kfda_class = coalesce(row.`식약처분류`, a.kfda_class),
                    a.indication1 = coalesce(row.`효능/효과1`, a.indication1),
                    a.indication2 = coalesce(row.`효능/효과2`, a.indication2),
                    a.target = coalesce(row.`대상`, a.target)
    WITH row, prod2, a
    OPTIONAL MATCH (b:Drug {name: trim(prod2)})
    WITH row, prod2, a, b
    CALL apoc.do.when(
      prod2 IS NULL OR trim(prod2) = "",
      'RETURN a AS a, b AS b, row AS row',
      'MERGE (b2:Drug {name: trim($prod2)})
       RETURN a AS a, b2 AS b, row AS row',
      {a:a, b:b, row:row, prod2:prod2}
    ) YIELD value
    WITH value.a AS a, value.b AS b, row
    MERGE (src:Source {name: coalesce(row.source_name, 'health.kr'), url: coalesce(row.source_url, '')})
      ON CREATE SET src.accessed_at = coalesce(row.accessed_at, date())
    FOREACH (_ IN CASE WHEN b IS NOT NULL THEN [1] ELSE [] END |
      MERGE (a)-[r:INTERACTS_WITH]->(b)
        ON CREATE SET r.severity = coalesce(row.`_severity_norm`, row.`결과`, ''),
                      r.note = coalesce(row.`사유`, ''),
                      r.source_id = id(src)
        ON MATCH SET  r.severity = coalesce(row.`_severity_norm`, r.severity),
                      r.note = coalesce(row.`사유`, r.note)
    )
    """
    rows = []
    for r in read_rows(args.csv):
        r["_severity_norm"] = normalize_severity(r.get("결과",""))
        if ("제품명2" not in r or not r.get("제품명2")) and r.get("제품명.1"):
            r["제품명2"] = r.get("제품명.1")
        rows.append(r)

    with driver.session() as s:
        BATCH = 1000
        for i in range(0, len(rows), BATCH):
            s.run(cypher, rows=rows[i:i+BATCH]).consume()

    driver.close()
    print(f"Ingested {len(rows)} rows.")

if __name__ == "__main__":
    main()
