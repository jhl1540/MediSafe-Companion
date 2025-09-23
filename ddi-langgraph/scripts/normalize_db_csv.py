\
import argparse, pandas as pd, os

EXPECTED_COLS = [
    "제품명","성분1","성분2","성분3","식약처분류",
    "효능/효과1","효능/효과2","대상","결과",
    "제품명2","사유"
]

def read_csv(path):
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError("Failed to read CSV with common encodings.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.getenv("DB_CSV", "/mnt/data/DB.csv"))
    args = ap.parse_args()

    df = read_csv(args.csv)
    if "제품명2" not in df.columns and "제품명.1" in df.columns:
        df = df.rename(columns={"제품명.1": "제품명2"})
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[EXPECTED_COLS]
    df.to_csv(args.csv, index=False, encoding="utf-8-sig")
    print(f"Normalized and saved: {args.csv}")

if __name__ == "__main__":
    main()
