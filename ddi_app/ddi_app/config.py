import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DB_CSV = os.getenv("DB_CSV", "./DB.csv")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

openai_client = OpenAI(api_key=OPENAI_API_KEY)