# neo4j_utils.py
from typing import List
from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def ensure_constraints(self):
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Component) REQUIRE c.name IS UNIQUE")

    def merge_components(self, drug: str, components: List[str]):
        if not components:
            return
        # Use real newlines (or spaces). No backslashes.
        q = """
        MERGE (d:Drug {name: $drug})
        WITH d
        UNWIND $components AS cname
        MERGE (c:Component {name: cname})
        MERGE (d)-[:HAS_COMPONENT]->(c)
        """
        with self.driver.session() as s:
            s.run(q, drug=drug, components=components)

    def merge_ddi(self, a: str, b: str, interaction: str, severity: str, source: str, confidence: float, evidence: str):
        q = """
        MERGE (a:Drug {name: $a})
        MERGE (b:Drug {name: $b})
        MERGE (a)-[r:INTERACTS_WITH]->(b)
        ON CREATE SET r.first_seen = timestamp()
        SET r.last_seen = timestamp(),
            r.description = $desc,
            r.severity    = $sev,
            r.source      = $src,
            r.confidence  = $conf,
            r.evidence    = $evid
        """
        q_reverse = """
        MERGE (a:Drug {name: $a})
        MERGE (b:Drug {name: $b})
        MERGE (b)-[r:INTERACTS_WITH]->(a)
        ON CREATE SET r.first_seen = timestamp()
        SET r.last_seen = timestamp(),
            r.description = $desc,
            r.severity    = $sev,
            r.source      = $src,
            r.confidence  = $conf,
            r.evidence    = $evid
        """
        with self.driver.session() as s:
            s.run(q, a=a, b=b, desc=interaction, sev=severity, src=source, conf=float(confidence or 0), evid=evidence)
            s.run(q_reverse, a=a, b=b, desc=interaction, sev=severity, src=source, conf=float(confidence or 0), evid=evidence)