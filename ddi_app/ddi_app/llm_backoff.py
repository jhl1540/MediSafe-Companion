import json
import asyncio
from typing import Optional, List
from pydantic import BaseModel, Field
from .config import openai_client

class DDIExtract(BaseModel):
    drug: str = Field(..., description="Drug name")
    components: List[str] = Field(default_factory=list)
    partner: Optional[str] = Field(None, description="Other drug if a pair query")
    interaction: Optional[str] = None
    severity: Optional[str] = None
    source: Optional[str] = None
    evidence: Optional[str] = None
    confidence: Optional[float] = 0.5

async def llm_extract_ddi(drug: str, partner: Optional[str]) -> List[DDIExtract]:
    system = "You are a clinical pharmacist. Return concise, factual JSON. Include `source` and short `evidence` text when you can."
    user = (f"Extract components and, if partner is given, the DDI between {drug}" +
            (f" and {partner}." if partner else ".") +
            " Return JSON array of objects with keys: drug, components, partner, interaction, severity, source, evidence, confidence.")

    def _call():
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    content = await asyncio.to_thread(_call)
    try:
        data = json.loads(content)
        arr = data if isinstance(data, list) else data.get("items") or data.get("data") or []
    except Exception:
        arr = []
    out: List[DDIExtract] = []
    for item in arr:
        try:
            out.append(DDIExtract(**item))
        except Exception:
            continue
    if not out:
        out.append(DDIExtract(drug=drug, partner=partner, components=[], interaction=None))
    return out