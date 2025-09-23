# graph_store.py
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv
import settings
import threading
_constraints_once = threading.Lock()
_constraints_done = False


@dataclass
class NeoCfg:
    uri: str = os.getenv("NEO4J_URI")
    user: str = os.getenv("NEO4J_USER")
    password: str = os.getenv("NEO4J_PASSWORD")
    
def _canon_pair(a: str, b: str):
    return (a, b) if a.lower() <= b.lower() else (b, a)    

class GraphStore:
    def __init__(self, cfg: NeoCfg | None = None):
        cfg = cfg or NeoCfg()
        self.driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))

    def ensure_constraints_once(self):
        global _constraints_done
        if _constraints_done:
            return
        with _constraints_once:
            if not _constraints_done:
                self.ensure_constraints()
                _constraints_done = True
       

    def ensure_constraints(self):
        q = """
        CREATE CONSTRAINT interaction_key IF NOT EXISTS
        FOR (i:Interaction) REQUIRE i.key IS UNIQUE;
        CREATE CONSTRAINT drug_name IF NOT EXISTS
        FOR (d:Drug) REQUIRE d.name IS UNIQUE;
        """
        with self.driver.session() as s:
            for stmt in [x.strip() for x in q.split(";") if x.strip()]:
                s.run(stmt)

    def upsert_drug(self, name: str, props: dict):
        q = """
        MERGE (d:Drug {name:$name})
        SET d += $props
        """
        params = {"name": name, "props": props or {}}
        with self.driver.session() as session:
            session.run(q, **params)      # session comes from self.driver.session()

    def upsert_interaction(self, a: str, b: str, severity: str, label: str,
                           mechanisms: list[str], refs: list[dict]):
        left, right = _canon_pair(a, b)
        key = f"{left}__{right}"
        q = """
        MERGE (la:Drug {name:$left})
        MERGE (lb:Drug {name:$right})
        MERGE (la)-[:INTERACTS_WITH]->(lb)

        MERGE (intr:Interaction {key:$key})
        SET intr.severity = coalesce($severity,''),
            intr.label    = coalesce($label,'interaction')

        WITH intr
        UNWIND $mechs AS m
          MERGE (mech:Mechanism {name:m})
          MERGE (intr)-[:VIA_MECHANISM]->(mech)
        WITH intr
        UNWIND $refs AS g
          MERGE (gd:Guideline {url:g.url})
          SET gd.title = coalesce(g.title, gd.title)
          MERGE (intr)-[:SUPPORTED_BY]->(gd)
        """
        with self.driver.session() as s:
            s.run(q, left=left, right=right, key=key,
                  severity=severity, mechs=mechanisms or [], refs=refs or [])

    def query_pair(self, a: str, b: str):
        q = """
        MATCH (a:Drug {name:$A})-[:INTERACTS_WITH]-(b:Drug {name:$B})
        RETURN a.name AS A, b.name AS B
        """
        with self.driver.session() as session:
            return [r.data() for r in session.run(q, A=a, B=b)]

    # Optional: one-time cleanup to remove duplicate reverse edges
    def migrate_make_canonical(self):
        fix = """
        // remove duplicate reverse edges keeping canonical direction
        MATCH (a:Drug)-[r:INTERACTS_WITH]->(b:Drug)
        WITH a,b,r,
             CASE WHEN toLower(a.name) <= toLower(b.name) THEN 1 ELSE 0 END AS ok
        WHERE ok = 0
        DELETE r
        """
        with self.driver.session() as s:
            s.run(fix)

