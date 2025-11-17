import openai
import os

# .env에 저장된 API 키 불러오기
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ 최신 SDK 방식 (openai >= 1.0.0)
from openai import OpenAI
client = OpenAI()

def check_available_models():
    try:
        models = client.models.list()
        print("✅ 사용 가능한 모델 목록:")
        for model in models.data:
            print("-", model.id)

        if any("gpt-4o" in model.id for model in models.data):
            print("\n🎉 이 API 키는 gpt-4o 모델을 사용할 수 있습니다!")
        else:
            print("\n❌ gpt-4o는 이 API 키에서 사용 불가합니다.")
            print("   👉 gpt-4 또는 gpt-4-turbo 사용을 고려하세요.")
    except Exception as e:
        print("❗ 오류 발생:", e)

# 실행
check_available_models()