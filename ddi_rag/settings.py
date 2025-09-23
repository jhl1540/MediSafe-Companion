# settings.py
import os
from pathlib import Path

try:
    # Load a specific path without find_dotenv() to avoid os.getcwd()
    dotenv_path = Path(__file__).resolve().parents[1] / ".env"  # project root/.env
    if dotenv_path.exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=dotenv_path, override=False)
except Exception:
    # If dotenv isn't installed or file missing, just continue (env can come from OS)
    pass
