import json
import asyncio

import pytest

import llm_backoff


@pytest.mark.asyncio
async def test_llm_extract_ddi_parses_json(monkeypatch):
    payload = {
        "items": [
            {
                "drug": "DrugA",
                "components": ["Comp1"],
                "partner": "DrugB",
                "interaction": "description",
                "severity": "moderate",
                "source": "guideline",
                "evidence": "ref",
                "confidence": 0.9,
            }
        ]
    }

    async def fake_to_thread(func, *args, **kwargs):
        return json.dumps(payload)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    extracts = await llm_backoff.llm_extract_ddi("DrugA", "DrugB")

    assert len(extracts) == 1
    assert extracts[0].interaction == "description"
    assert extracts[0].severity == "moderate"
    assert extracts[0].confidence == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_llm_extract_ddi_returns_fallback_on_parse_error(monkeypatch):
    async def fake_to_thread(func, *args, **kwargs):
        return "not-json"

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    extracts = await llm_backoff.llm_extract_ddi("DrugA", None)

    assert len(extracts) == 1
    assert extracts[0].drug == "DrugA"
    assert extracts[0].partner is None
    assert extracts[0].interaction is None
