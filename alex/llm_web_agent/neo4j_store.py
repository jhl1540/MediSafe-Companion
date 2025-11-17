# n
# Helper for Neo4j graph operations (queries, logging, verification)

import os
from typing import Optional, Dict, Any, List
from datetime import datetime

from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth

load_dotenv()

NEO4J_URI       = os.getenv("NEO4J_URI")
NEO4J_USER      = os.getenv("NEO4J_USER")
NEO4J_PASSWORD  = os.getenv("NEO4J_PASSWORD")
NEO4J_API_KEY   = os.getenv("NEO4J_API_KEY")  # optional (Aura)
NEO4J_DATABASE  = os.getenv("NEO4J_DATABASE")  # optional

def _make_auth(user: Optional[str], password: Optional[str], api_key: Optional[str]):
    if user and password:
        return basic_auth(user, password)
    if api_key:
        try:
            from neo4j.auth_management import BearerAuth
            return BearerAuth(api_key)
        except Exception:
            raise RuntimeError(
                "NEO4J_API_KEY is set but BearerAuth is unavailable. "
                "Use neo4j>=5.26 or set NEO4J_USER/NEO4J_PASSWORD."
            )
    raise RuntimeError("Missing credentials. Set NEO4J_USER/NEO4J_PASSWORD or NEO4J_API_KEY.")

class GraphStore:
    def __init__(self,
                 uri: Optional[str] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 api_key: Optional[str] = None,
                 database: Optional[str] = None):
        uri      = uri or NEO4J_URI
        user     = user if user is not None else NEO4J_USER
        password = password if password is not None else NEO4J_PASSWORD
        api_key  = api_key  if api_key  is not None else NEO4J_API_KEY
        database = database if database is not None else NEO4J_DATABASE

        if not uri:
            raise RuntimeError("NEO4J_URI is not set.")
        auth = _make_auth(user, password, api_key)
        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._database = database

        try:
            with self._driver.session(database=self._database) as s:
                s.run("RETURN 1 AS ok").single()
        except Exception as e:
            raise RuntimeError(f"Neo4j connectivity failed: {e}") from e

    def close(self):
        self._driver.close()

    def ensure_schema(self) -> None:
        cyphers = [
            "CREATE CONSTRAINT patient_id_unique IF NOT EXISTS FOR (p:Patient)     REQUIRE p.patient_id IS UNIQUE",
            "CREATE CONSTRAINT drug_name_unique   IF NOT EXISTS FOR (d:Drug)        REQUIRE d.name       IS UNIQUE",
            "CREATE CONSTRAINT ingr_name_unique   IF NOT EXISTS FOR (i:Ingredient)  REQUIRE i.name       IS UNIQUE",
            "CREATE CONSTRAINT query_id_unique    IF NOT EXISTS FOR (q:Query)       REQUIRE q.id         IS UNIQUE",
            "CREATE INDEX     drug_display_name   IF NOT EXISTS FOR (d:Drug)        ON (d.display_name)"
        ]
        try:
            with self._driver.session(database=self._database) as s:
                for c in cyphers:
                    s.run(c)
                try:
                    names = [r["name"] for r in s.run("CALL db.indexes()")]
                    if "drug_fulltext" not in names:
                        s.run("""
                        CALL db.index.fulltext.createNodeIndex(
                          'drug_fulltext', ['Drug'], ['name','display_name']
                        )
                        """)
                except Exception:
                    pass
        except Exception:
            pass

    def upsert_drug(self, display_name: str, ingredient_text: Optional[str] = None) -> Dict[str, Any]:
        name_norm = display_name.strip().lower()
        ingredients = []
        if ingredient_text:
            ingredients = [p.strip() for p in ingredient_text.replace("/", ",").split(",") if p.strip()]
        cypher = """
        MERGE (d:Drug {name:$name_norm})
        ON CREATE SET d.display_name = $display_name, d.createdAt = datetime()
        SET d.updatedAt = datetime()
        WITH d
        FOREACH(ing IN $ingredients |
          MERGE (i:Ingredient {name:toLower(ing)})
          MERGE (d)-[:CONTAINS_INGREDIENT]->(i)
        )
        RETURN d{.*} AS drug
        """
        with self._driver.session(database=self._database) as s:
            rec = s.run(cypher, name_norm=name_norm, display_name=display_name, ingredients=ingredients).single()
            return rec["drug"]

    def log_query_and_result(self, *, user_id: str, text: str,
                             drug1_display: str, drug2_display: Optional[str],
                             sections: Dict[str, Any]) -> str:
        d1_key = drug1_display.strip().lower()
        d2_key = (drug2_display or "").strip().lower() or None
        now = datetime.utcnow().isoformat()
        qkey = f"{user_id}:{d1_key}:{d2_key or ''}"
        cypher = """
        MERGE (u:Patient {patient_id:$user_id})
          ON CREATE SET u.createdAt = datetime()
        SET u.updatedAt = datetime()

        MERGE (d1:Drug {name:$d1_key})
          ON CREATE SET d1.display_name = $drug1_display, d1.createdAt = datetime()
        SET  d1.updatedAt = datetime($now),
            d1.card       = $drug1_card,
            d1.ingredient = $ingredient1,
            d1.warn_url   = $warn_url1

        MERGE (q:Query {id:$qkey})
          ON CREATE SET q.createdAt = datetime()
        SET  q.mode = CASE WHEN $d2_key IS NULL THEN 'single' ELSE 'pair' END,
            q.ts   = datetime($now)

        MERGE (q)-[:ASKED_BY]->(u)
        MERGE (q)-[:ABOUT]->(d1)

        // drug2가 있을 때만 실행
        FOREACH (_ IN CASE WHEN $d2_key IS NULL THEN [] ELSE [1] END |
          MERGE (d2:Drug {name:$d2_key})
            ON CREATE SET d2.display_name = $drug2_display, d2.createdAt = datetime()
          SET  d2.updatedAt  = datetime($now),
              d2.card       = $drug2_card,
              d2.ingredient = $ingredient2,
              d2.warn_url   = $warn_url2

          MERGE (q)-[:ABOUT_SECOND]->(d2)

          // 상호작용 엣지에 페이로드 저장 (단방향이면 충분: 조회는 -[]- 무방향으로 함)
          MERGE (d1)-[i:INTERACTS_WITH]->(d2)
            ON CREATE SET i.first_seen = date(), i.evidence_qids = []
          SET  i.interaction_md = $interaction_md,
              i.last_text      = $text,
              i.last_seen      = date(),
              i.evidence_qids  = CASE
                                    WHEN i.evidence_qids IS NULL OR NOT $qkey IN i.evidence_qids
                                      THEN coalesce(i.evidence_qids, []) + [$qkey]
                                    ELSE i.evidence_qids
                                  END
        )

        RETURN q.id AS qid
        """
        params = {
            "user_id": user_id, "qkey": qkey, "now": now,
            "d1_key": d1_key, "d2_key": d2_key,
            "drug1_display": drug1_display, "drug2_display": drug2_display,
            "drug1_card":  sections.get("drug1_card", ""),
            "drug2_card":  sections.get("drug2_card", ""),
            "ingredient1": sections.get("ingredient1", ""),
            "ingredient2": sections.get("ingredient2", ""),
            "warn_url1":   sections.get("warn_url1", ""),
            "warn_url2":   sections.get("warn_url2", ""),
            "interaction_md": sections.get("interaction_md", ""),
            "text": text,
        }
        with self._driver.session(database=self._database) as s:
            return s.run(cypher, **params).single()["qid"]

    def upsert_verification(self, a_name: str, b_name: str,
                            status: str, summary: str, sources: List[str]):
        a = a_name.strip().lower()
        b = b_name.strip().lower()
        cy = """
        MERGE (a:Drug {name:$a}) ON CREATE SET a.display_name=$a_disp, a.createdAt=datetime()
        MERGE (b:Drug {name:$b}) ON CREATE SET b.display_name=$b_disp, b.createdAt=datetime()
        MERGE (a)-[i1:INTERACTS_WITH]->(b)
          ON CREATE SET i1.first_seen = date()
        SET i1.verify_status  = $status,
            i1.verify_summary = $summary,
            i1.verify_sources = $sources,
            i1.verify_ts      = date()
        MERGE (b)-[i2:INTERACTS_WITH]->(a)
          ON CREATE SET i2.first_seen = date()
        SET i2.verify_status  = $status,
            i2.verify_summary = $summary,
            i2.verify_sources = $sources,
            i2.verify_ts      = date()
        """
        with self._driver.session(database=self._database) as s:
            s.run(cy, a=a, b=b, a_disp=a_name, b_disp=b_name,
                  status=status, summary=summary, sources=sources)

    def resolve_drug_name(self, query_text: str) -> Optional[Dict[str, Any]]:
        q = (query_text or "").strip()
        if not q:
            return None
        cy = """
        CALL {
          WITH toLower($q) AS k
          MATCH (d:Drug)
          WHERE d.name = k
             OR toLower(d.display_name) CONTAINS k
             OR (d.synonyms IS NOT NULL AND any(s IN d.synonyms WHERE toLower(s) CONTAINS k))
          RETURN d, 1.0 AS score
          LIMIT 1
        }
        RETURN d, score
        UNION
        CALL {
          WITH $q AS q
          CALL db.index.fulltext.queryNodes('drug_fulltext', q+'~') YIELD node, score
          RETURN node AS d, score
          LIMIT 1
        }
        RETURN d, score
        ORDER BY score DESC
        LIMIT 1
        """
        try:
            with self._driver.session(database=self._database) as s:
                row = s.run(cy, q=q).single()
        except Exception:
            cy_simple = """
            WITH toLower($q) AS k
            MATCH (d:Drug)
            WHERE d.name = k
               OR toLower(d.display_name) CONTAINS k
               OR (d.synonyms IS NOT NULL AND any(s IN d.synonyms WHERE toLower(s) CONTAINS k))
            RETURN d
            LIMIT 1
            """
            with self._driver.session(database=self._database) as s:
                row = s.run(cy_simple, q=q).single()
        if not row:
            return None
        d = row.get("d")
        return {"name": d.get("name"), "display_name": d.get("display_name")}

    def find_interactions_for_drug(self, drug_name_or_alias: str):
        key = (drug_name_or_alias or "").strip().lower()
        cypher = """
        MATCH (d:Drug)
        WHERE d.name = $key OR toLower(d.display_name) CONTAINS $key
        WITH d
        MATCH (d)-[i:INTERACTS_WITH]-(other:Drug)
        WITH d, other, i
        ORDER BY other.display_name, coalesce(i.last_seen, date('1900-01-01')) DESC
        WITH d, other, head(collect(i)) AS i
        RETURN
            d.display_name                 AS drug,
            other.display_name             AS interacts_with,
            coalesce(i.interaction_md,'')  AS interaction_md,
            coalesce(i.severity,'Unknown') AS severity,
            coalesce(i.mechanism,'')       AS mechanism,
            coalesce(i.management,'')      AS management,
            coalesce(i.source,'')          AS source,
            i.last_seen                    AS last_seen,
            coalesce(i.verify_status,'')   AS verify_status,
            coalesce(i.verify_summary,'')  AS verify_summary,
            coalesce(i.verify_sources,[])  AS verify_sources,
            i.verify_ts                    AS verify_ts
        ORDER BY last_seen DESC, interacts_with
        """
        with self._driver.session(database=self._database) as s:
            return [dict(r) for r in s.run(cypher, key=key)]

    def get_chunks_for_drug(self, drug: str, k: int = 8) -> List[Dict[str, Any]]:
        key = (drug or "").strip().lower()
        cy = """
        MATCH (d:Drug)
        WHERE d.name = $key OR toLower(d.display_name) CONTAINS $key
        WITH d
        MATCH (c:Chunk)-[:MENTIONS]->(d)
        OPTIONAL MATCH (doc:Document)-[:HAS_CHUNK]->(c)
        RETURN c.chunk_id AS chunk_id, c.text AS text,
               doc.title AS title, doc.source_url AS source_url
        LIMIT $k
        """
        with self._driver.session(database=self._database) as s:
            return [dict(r) for r in s.run(cy, key=key, k=k)]

    def get_drug_node(self, drug: str) -> Optional[Dict[str, Any]]:
        key = (drug or "").strip().lower()
        cy = """
        MATCH (d:Drug)
        WHERE d.name = $key OR toLower(d.display_name) CONTAINS $key
        RETURN d{.*} AS d
        LIMIT 1
        """
        with self._driver.session(database=self._database) as s:
            r = s.run(cy, key=key).single()
            return r["d"] if r else None

    def get_user_history(self, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        cy = """
        MATCH (q:Query)-[:ASKED_BY]->(u:Patient {patient_id:$uid})
        OPTIONAL MATCH (q)-[:ABOUT]->(d1:Drug)
        OPTIONAL MATCH (q)-[:ABOUT_SECOND]->(d2:Drug)
        OPTIONAL MATCH (d1)-[i:INTERACTS_WITH]->(d2)
        WITH q, u, d1, d2, i
        RETURN q.id AS qid, q.mode AS mode, q.ts AS ts,
               d1.display_name AS drug1, d2.display_name AS drug2,
               coalesce(i.interaction_md,'') AS interaction_md
        ORDER BY q.ts DESC
        LIMIT $limit
        """
        with self._driver.session(database=self._database) as s:
            return [dict(r) for r in s.run(cy, uid=user_id, limit=limit)]