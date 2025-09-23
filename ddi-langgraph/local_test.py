# local_test.py
import os
from langchain_core.messages import HumanMessage
from graph import app
from dotenv import load_dotenv
load_dotenv()


os.environ.setdefault("DB_CSV", "./DB.csv")

print("Single-drug test: '타이레놀'")
res1 = app.invoke({"messages":[HumanMessage(content="타이레놀")]}, config={})
print(res1["messages"][-1]["content"])

print("\nTwo-drug test: '로수젯정 타이레놀'")
res2 = app.invoke({"messages":[HumanMessage(content="로수젯정 타이레놀")]}, config={})
print(res2["messages"][-1]["content"])
