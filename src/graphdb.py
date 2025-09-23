from __future__ import annotations
from typing import Any, Dict, List

from .utils import simple_embedding


class InMemoryGraphDB:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.interactions: List[Dict[str, Any]] = []
        self.query_records: List[Dict[str, Any]] = []
        self.brand_records: List[Dict[str, Any]] = []

    def upsert_drug(self, drug: Dict[str, Any]):
        key = drug.get("inchikey") or drug.get("id") or drug.get("name")
        if not key:
            raise ValueError("Drug node missing identifier")
        existing = self.nodes.get(key, {})
        merged = {**existing, **drug}
        self.nodes[key] = merged
        return key

    def list_drugs(self) -> List[Dict[str, Any]]:
        return list(self.nodes.values())

    def add_interaction(self, d1: str, d2: str, severity: str, mechanism: str, refs: List[str]):
        self.interactions.append({
            "drug1": d1,
            "drug2": d2,
            "severity": severity,
            "mechanism": mechanism,
            "refs": refs,
        })

    def get_interactions_for(self, inchikey: str) -> List[Dict[str, Any]]:
        return [e for e in self.interactions if e["drug1"] == inchikey or e["drug2"] == inchikey]

    def add_query_record(self, record: Dict[str, Any]):
        self.query_records.append(record)

    def add_brand_record(self, record: Dict[str, Any]):
        self.brand_records.append(record)

    def ensure_demo_seed(self):
        acet = {
            "name": "Acetaminophen",
            "synonyms": ["Tylenol", "Paracetamol"],
            "smiles": "CC(=O)NC1=CC=C(O)C=C1O",
            "inchikey": "RZVAJINKPMORJF-UHFFFAOYSA-N",
            "class": "Analgesic",
            "embedding": simple_embedding("Acetaminophen"),
        }
        ethanol = {
            "name": "Ethanol",
            "synonyms": ["Alcohol"],
            "smiles": "CCO",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "class": "CNS Depressant",
            "embedding": simple_embedding("Ethanol"),
        }
        salicylic = {
            "name": "Salicylic acid",
            "synonyms": ["2-Hydroxybenzoic acid"],
            "smiles": "C1=CC(=CC=C1C(=O)O)O",
            "inchikey": "YGSDEFSMJLZEOE-UHFFFAOYSA-N",
            "class": "NSAID",
            "embedding": simple_embedding("Salicylic acid"),
        }
        amyl_nitrite = {
            "name": "Amyl nitrite",
            "synonyms": ["Pentyl nitrite"],
            "smiles": "CCCCCONO",
            "inchikey": "CWHHHXQLPUJAES-UHFFFAOYSA-N",
            "class": "Vasodilator",
            "embedding": simple_embedding("Amyl nitrite"),
        }
        for d in (acet, ethanol, salicylic, amyl_nitrite):
            self.upsert_drug(d)

        self.add_interaction(
            d1=acet["inchikey"],
            d2=ethanol["inchikey"],
            severity="High",
            mechanism=(
                "CYP2E1 induction by ethanol increases NAPQI formation from "
                "acetaminophen → hepatotoxicity"
            ),
            refs=["DDInter:DDInter14", "NIAAA guidance"],
        )


# Export a shared instance
GRAPH = InMemoryGraphDB()
GRAPH.ensure_demo_seed()