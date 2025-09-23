

from __future__ import annotations
import hashlib
import json
from typing import Any, List
import numpy as np


def simple_embedding(text: str, dim: int = 128) -> List[float]:
    """Deterministic toy embedding (hash → pseudo-random). Replace with a model."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand to dim via repeated hashing
    bytes_needed = dim
    buf = bytearray()
    seed = h
    while len(buf) < bytes_needed:
        seed = hashlib.sha256(seed).digest()
        buf.extend(seed)
    arr = np.frombuffer(bytes(buf[:dim]), dtype=np.uint8).astype(np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr.tolist()


def cosine_sim(a: List[float], b: List[float]) -> float:
    import numpy as _np
    va, vb = _np.array(a), _np.array(b)
    denom = (float(_np.linalg.norm(va)) * float(_np.linalg.norm(vb))) + 1e-9
    return float(_np.dot(va, vb) / denom)


def _safe_json_parse(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
        return {}