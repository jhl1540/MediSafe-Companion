\
import os
from typing import Dict, Any
from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(URI, auth=AUTH)
    def close(self):
        self.driver.close()

    def single_info(self, drug: str) -> Dict[str, Any]:
        cypher = """
        MATCH (d:Drug)
        WHERE toLower(d.name) = toLower($drug)
        RETURN d.name AS 제품명,
               d.ingredient1 AS 성분1, d.ingredient2 AS 성분2, d.ingredient3 AS 성분3,
               d.kfda_class AS 식약처분류, d.indication1 AS `효능/효과1`, d.indication2 AS `효능/효과2`,
               d.target AS 대상,
               '' AS 결과
        LIMIT 1
        """
        with self.driver.session() as s:
            rec = s.run(cypher, drug=drug).single()
            return rec.data() if rec else {}

    def pair_info(self, drug_a: str, drug_b: str) -> Dict[str, Any]:
        cypher = """
        MATCH (a:Drug) WHERE toLower(a.name)=toLower($a)
        MATCH (b:Drug) WHERE toLower(b.name)=toLower($b)
        MATCH (a)-[r:INTERACTS_WITH]->(b)
        OPTIONAL MATCH (src) WHERE id(src) = r.source_id
        RETURN a.name AS 제품명1,
               b.name AS 제품명2,
               r.note AS 사유,
               r.severity AS 결과,
               a.ingredient1 AS 성분1, a.ingredient2 AS 성분2, a.ingredient3 AS 성분3,
               a.kfda_class AS 식약처분류, a.indication1 AS `효능/효과1`, a.indication2 AS `효능/효과2`,
               a.target AS 대상
        UNION
        MATCH (b:Drug) WHERE toLower(b.name)=toLower($a)
        MATCH (a:Drug) WHERE toLower(a.name)=toLower($b)
        MATCH (a)-[r:INTERACTS_WITH]->(b)
        OPTIONAL MATCH (src) WHERE id(src) = r.source_id
        RETURN a.name AS 제품명1,
               b.name AS 제품명2,
               r.note AS 사유,
               r.severity AS 결과,
               a.ingredient1 AS 성분1, a.ingredient2 AS 성분2, a.ingredient3 AS 성분3,
               a.kfda_class AS 식약처분류, a.indication1 AS `효능/효과1`, a.indication2 AS `효능/효과2`,
               a.target AS 대상
        LIMIT 1
        """
        with self.driver.session() as s:
            rec = s.run(cypher, a=drug_a, b=drug_b).single()
            return rec.data() if rec else {}
