from __future__ import annotations
from typing import Any, Dict, List, Optional

# Try real LangGraph. If unavailable (e.g., sandbox), fall back to a tiny runner
LANGGRAPH_AVAILABLE = False
END = "__END__"

try:
    from langgraph.graph import StateGraph as _LGStateGraph, END as _LG_END  # type: ignore
    LANGGRAPH_AVAILABLE = True
    END = _LG_END
except Exception:
    LANGGRAPH_AVAILABLE = False


class _MiniStateGraph:
    """Minimal orchestrator compatible with .add_node/.add_edge/.compile()."""

    def __init__(self, _state_type):
        self._nodes: Dict[str, Any] = {}
        self._edges: Dict[str, List[str]] = {}
        self._entry: Optional[str] = None

    def add_node(self, name: str, fn):
        self._nodes[name] = fn

    def set_entry_point(self, name: str):
        self._entry = name

    def add_edge(self, src: str, dst: str):
        self._edges.setdefault(src, []).append(dst)

    def compile(self):
        graph = self

        class _Runner:
            def invoke(self, state: Dict[str, Any]):
                if graph._entry is None:
                    raise RuntimeError("No entry point set")
                cur = graph._entry
                s = dict(state)
                steps = 0
                while cur and cur != END:
                    if cur not in graph._nodes:
                        raise RuntimeError(f"Node '{cur}' not found")
                    fn = graph._nodes[cur]
                    res = fn(s)
                    if isinstance(res, dict):
                        s = res
                    nxts = graph._edges.get(cur, [])
                    if not nxts:
                        break
                    cur = nxts[0]  # deterministic path
                    steps += 1
                    if steps > 10000:
                        raise RuntimeError("Graph appears to loop")
                return s

        return _Runner()


StateGraph = _LGStateGraph if LANGGRAPH_AVAILABLE else _MiniStateGraph