import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from rapidfuzz import process, fuzz

# Load .env environment
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o")
websearch = TavilySearch()

def canonicalize_drug_name(user_input, drug_names):
    """Use fuzzy search and LLM to canonicalize user drug input."""
    # First try fuzzy search
    best, score, _ = process.extractOne(user_input, drug_names, scorer=fuzz.QRatio)
    if score >= 85:      # Tune threshold as needed
        return best

    # Otherwise, use LLM for abbreviation/typo transliteration/translation
    prompt = (
        f"Find the most likely match for the drug name '{user_input}' from this list: "
        f"{', '.join(drug_names)}. Return only the closest match, as it would appear in the DB."
    )
    llm_result = llm.predict(prompt).strip()
    # Use the closest fuzzy match from LLM answer for safety
    if llm_result in drug_names:
        return llm_result
    # fallback/fuzzy again
    best2, score2, _ = process.extractOne(llm_result, drug_names, scorer=fuzz.QRatio)
    if score2 > 70:
        return best2
    return None

def parse_and_canonicalize_node(state):
    db = pd.read_csv('DB.csv')
    names1 = set(db['제품명1'].dropna().astype(str))
    names2 = set(db['제품명2'].dropna().astype(str))
    all_names = names1 | names2

    state['drug1'] = canonicalize_drug_name(state['user_input1'], all_names)
    state['drug2'] = canonicalize_drug_name(state['user_input2'], all_names) if state.get('user_input2') else None

    if not state['drug1'] or (state.get('user_input2') and not state['drug2']):
        state['error'] = "One or both drug names could not be matched. Please check your inputs."
    return state

def db_or_web_lookup_node(state):
    db = pd.read_csv('DB.csv')
    drug1 = state.get('drug1')
    drug2 = state.get('drug2')

    result = None
    # Order-independent search for two-drug interaction
    if drug1 and drug2:
        mask = ((db['제품명1'] == drug1) & (db['제품명2'] == drug2)) | \
               ((db['제품명1'] == drug2) & (db['제품명2'] == drug1))
        rows = db[mask]
        state['source'] = "db"
    # Single drug lookup
    elif drug1:
        mask = (db['제품명1'] == drug1)
        rows = db[mask]
        state['source'] = "db"
    else:
        rows = pd.DataFrame()

    if not rows.empty:
        result = rows.to_dict(orient='records')
    # Fallback to LLM+Web if nothing found
    else:
        query = f"Drug info for {drug1}" if (drug1 and not drug2) else f"Interaction between {drug1} and {drug2}"
        tavily_result = websearch.invoke({"query": query})
        llm_resp = llm.predict(
            f"Based on this context: '{tavily_result}', summarize the relevant drug information or interaction (clear, concise, Korean summary)."
        )
        result = [{"info": llm_resp, "source": "web"}]
        state['source'] = "web"
    state['result'] = result
    if not state['result']:
        state['error'] = "No matching entry in database or web."
    return state


def store_result_and_init_node(state):
    db = pd.read_csv('DB.csv')
    record = {}
    # For drug interaction: drugs + result
    if state.get('drug1') and state.get('drug2'):
        record = {
            '제품명1': state['drug1'],
            '제품명2': state['drug2'],
            # If DB result, preserve known columns. If web result, fill new columns as suitable.
            '결과': state['result'][0].get('결과', '') if state['source'] == "db" else '',
            '사유': state['result'][0].get('사유', '') if state['source'] == "db" else state['result'][0].get('info', ''),
            # More columns can be added here...
        }
    # For single drug
    elif state.get('drug1'):
        record = {
            '제품명1': state['drug1'],
            '제품명2': '',
            '결과': state['result'][0].get('결과', '') if state['source'] == "db" else '',
            '사유': state['result'][0].get('사유', '') if state['source'] == "db" else state['result'][0].get('info', ''),
            # More columns e.g. 효능/효과... could be parsed and stored
        }
    # Append row only if both drug or one drug was recognized
    if state.get('drug1'):
        db = pd.concat([db, pd.DataFrame([record])], ignore_index=True)
        db.to_csv('DB.csv', index=False)
    # Reset state for next query
    return {
        'user_input1': '',
        'user_input2': '',
        'drug1': None,
        'drug2': None,
        'result': None,
        'source': None,
        'error': None
    }