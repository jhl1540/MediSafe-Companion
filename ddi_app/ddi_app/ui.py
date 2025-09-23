from typing import Dict, Any, List, Optional

def monograph_md(m: Dict[str, Any]) -> str:
    comps = ", ".join(m.get("components", []) or []) or "정보 없음"
    return (
        f"**구성 성분:** {comps}\\n\\n"
        f"**DB 내 상호작용 레코드 수:** {len(m.get('interactions', []))}\\n"
    )

def format_answer(drug1: str, mono1: Dict[str, Any], drug2: Optional[str], mono2: Optional[Dict[str, Any]], ddi: Optional[Dict[str, Any]]) -> str:
    sections: List[str] = []
    sections.append(f"### 📌 약물 1: {drug1}\\n\\n" + monograph_md(mono1 or {}))
    if drug2 and mono2:
        sections.append(f"### 📌 약물 2: {drug2}\\n\\n" + monograph_md(mono2))

    if drug2:
        sections.append("### 💥 두 약물의 상호작용")
        if ddi and ddi.get("interaction"):
            src = ddi.get("source","DB/LLM/WEB")
            conf = ddi.get("confidence")
            ev = ddi.get("evidence")
            sections.append(
                f"- ✅ **함께 복용 가능 여부/주의:** {ddi.get('severity','미상')}\\n"
                f"- ❗ **요약:** {ddi.get('interaction')}\\n"
                f"- 📚 **출처:** {src}  |  **신뢰도:** {conf if conf is not None else 'N/A'}\\n"
                + (f"- 🔎 **근거:** {ev}" if ev else "")
            )
        else:
            sections.append("- DB/웹에서 명확한 상호작용 정보를 찾지 못했습니다. 최신 자료를 기반으로 추가 확인이 필요합니다.")
    return "\n\n".join(sections)