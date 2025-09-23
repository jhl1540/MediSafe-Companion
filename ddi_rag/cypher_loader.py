# cypher_loader.py
from pathlib import Path
BASE = Path(__file__).parent / "cypher"
def load(name: str) -> str:
    return (BASE / name).read_text(encoding="utf-8")
