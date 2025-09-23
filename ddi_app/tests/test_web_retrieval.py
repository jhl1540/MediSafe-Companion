import pytest
from typing import Optional

import web_retrieval
from web_retrieval import WebExtract


@pytest.mark.asyncio
async def test_web_retrieve_prefers_highest_confidence(monkeypatch):
    async def fake_health_kr(drug: str):
        return [
            WebExtract(drug=drug, partner="DrugB", interaction="low", confidence=0.55, source="health.kr"),
            WebExtract(drug=drug, partner="DrugB", interaction="high", confidence=0.72, source="health.kr"),
            WebExtract(drug=drug, components=["Comp1"], confidence=0.6, source="health.kr"),
        ]

    async def fake_ddinter(drug: str, partner: Optional[str]):
        return [
            WebExtract(drug=drug, partner="DrugB", interaction="ddinter", confidence=0.8, source="ddinter"),
        ]

    monkeypatch.setattr(web_retrieval, "fetch_health_kr", fake_health_kr)
    monkeypatch.setattr(web_retrieval, "fetch_ddinter", fake_ddinter)

    results = await web_retrieval.web_retrieve("DrugA", "DrugB")

    assert len(results) == 2  # one for the pair, one for components only

    pair_extract = next(item for item in results if item.partner)
    assert pair_extract.interaction == "ddinter"
    assert pair_extract.confidence == pytest.approx(0.8)

    mono_extract = next(item for item in results if not item.partner)
    assert mono_extract.components == ["Comp1"]
