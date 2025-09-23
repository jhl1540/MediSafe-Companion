
# ddi_gradio_app.py
import os
import gradio as gr
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from parsers import search_healthkr_drug_cd, fetch_healthkr_fields, parse_ddi2_pair
from graph_store import GraphStore, NeoCfg
from db_io import load_db, save_db, upsert_single, upsert_pair, DB_COLUMNS

# Load .env early
load_dotenv(find_dotenv(), override=False)

DB_PATH = Path("data/DB.csv")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

USE_GRAPH = True
neo = GraphStore() if USE_GRAPH else None   # <- env-backed config

COLUMNS = ['제품명1','성분1','성분2','성분3','식약처분류','효능/효과1','효능/효과2','대상','결과','제품명2','사유']

def load_db() -> pd.DataFrame:
    if DB_PATH.exists():
        return pd.read_csv(DB_PATH)
    return pd.DataFrame(columns=COLUMNS)

def save_db(df):
    df.to_csv(DB_PATH, index=False)


def enrich_single(prod: str):
    cd = search_healthkr_drug_cd(prod)
    h = fetch_healthkr_fields(cd) if cd else None
    if not h:
        return {}
    # 효능효과_md → 효능/효과1,2
    eff1, eff2 = "", ""
    if h.효능효과_md:
        lines = [l[2:].strip() if l.startswith("- ") else l for l in h.효능효과_md.splitlines() if l.strip()]
        eff1 = lines[0] if lines else ""
        eff2 = lines[1] if len(lines) > 1 else ""
    return {
        '성분1': h.성분1, '성분2': h.성분2, '성분3': h.성분3,
        '식약처분류': h.식약처분류, '효능/효과1': eff1, '효능/효과2': eff2, '대상': h.대상
    }


def on_submit(drug1: str, drug2: str, write_back: bool):
    drug1 = (drug1 or "").strip()
    drug2 = (drug2 or "").strip()
    if not drug1 and not drug2:
        return load_db().tail(10), "제품명을 입력해 주세요"

    df = load_db()

    # 1) 단일 보강
    if drug1 and not drug2:
        base = {c: "" for c in COLUMNS}
        base['제품명1'] = drug1
        upd = enrich_single(drug1)
        base.update(upd)
        # upsert
        mask = (df['제품명1'].fillna('').str.lower()==drug1.lower()) & (df['제품명2'].fillna('')=='')
        if mask.any():
            df.loc[mask, list(upd.keys())] = pd.Series(upd)
        else:
            df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)

        if USE_GRAPH and neo:
            neo.upsert_drug(drug1, upd)

        msg = f"'{drug1}' 정보 보강 완료"

    # 2) 두 약물 보강(사유/결과)
    if drug1 and drug2:
        # 개별 보강
        for d in [drug1, drug2]:
            upd = enrich_single(d)
            if USE_GRAPH and neo:
                neo.upsert_drug(d, upd)
            mask = (df['제품명1'].fillna('').str.lower()==d.lower()) & (df['제품명2'].fillna('')=='')
            base = {c: "" for c in COLUMNS}
            base['제품명1'] = d
            base.update(upd)
            if mask.any():
                df.loc[mask, list(upd.keys())] = pd.Series(upd)
            else:
                df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)

        # 상호작용
        ddi = parse_ddi2_pair(drug1, drug2)
        pair = {c: "" for c in COLUMNS}
        pair['제품명1'] = drug1
        pair['제품명2'] = drug2
        pair['결과'] = ddi.결과
        pair['사유'] = ddi.사유
        mask = (df['제품명1'].fillna('').str.lower()==drug1.lower()) & (df['제품명2'].fillna('').str.lower()==drug2.lower())
        if mask.any():
            df.loc[mask, ['결과','사유']] = [ddi.결과, ddi.사유]
        else:
            df = pd.concat([df, pd.DataFrame([pair])], ignore_index=True)

        if USE_GRAPH and neo:
            neo.upsert_interaction(drug1, drug2, severity=ddi.중증도, label='interaction',
                                   mechanisms=ddi.메커니즘 or [], refs=ddi.근거문헌 or [])

        msg = f"'{drug1} + {drug2}' 상호작용(사유/결과) 보강 완료"

    if write_back:
        save_db(df)
        msg += ", DB 저장 완료"

    return df.tail(10), msg

with gr.Blocks(title="DDI RAG 통합") as demo:
    gr.Markdown("### 사용자 처방 DB 업데이트 + health.kr + DDInter 2.0 + Neo4j")
    with gr.Row():
        drug1 = gr.Textbox(label="제품명1", placeholder="예: 크레스토정 10mg")
        drug2 = gr.Textbox(label="제품명2 (선택)", placeholder="예: 클라리스로마이신")
    write_back = gr.Checkbox(value=True, label="DB.csv 즉시 저장")
    submit = gr.Button("조회 및 업데이트")
    out_df = gr.Dataframe(label="미리보기 (최근 10행)")
    out_msg = gr.Textbox(label="상태")
    submit.click(on_submit, [drug1,drug2,write_back], [out_df,out_msg])

if __name__ == "__main__":
    demo.launch()

## 환경 변수 & 실행
'''


# Neo4j 로컬 예시
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=yourpassword

# Python deps (예)
pip install requests beautifulsoup4 neo4j langgraph gradio pydantic

# 실행
python ddi_gradio_app.py
# or
python -m langgraph_pipeline  # 테스트 실행

'''


