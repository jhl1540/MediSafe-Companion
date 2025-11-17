
# Placeholder extractor (replace with your LLM-powered extractor)
# Return structure: {"drugs": ["name1", "name2", ...]}

import re

def extract_drugs(text: str):
    toks = set()
    for w in re.findall(r"[A-Za-z가-힣0-9]+", text):
        if len(w) >= 3 and (w[0].isupper() or w.endswith(("정","캡슐","산","핀"))):
            toks.add(w)
    return {"drugs": list(toks)}
