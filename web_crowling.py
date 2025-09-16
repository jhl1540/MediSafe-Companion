import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_drug_interaction_info(url):
    """
    주어진 URL에서 약물 상호작용 정보를 스크래핑하여 
    '성분1', '성분2', '내용'으로 구성된 데이터베이스(DataFrame)를 생성합니다.
    """
    
    # 💡 [수정된 부분] 브라우저처럼 보이기 위한 헤더 정보
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # 💡 [수정된 부분] 헤더 정보를 포함하여 요청
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"웹 페이지를 가져오는 중 오류가 발생했습니다: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    
    interaction_table = soup.find('table', class_='result_table3')

    if not interaction_table:
        print("상호작용 정보 테이블을 찾을 수 없습니다.")
        return None

    interaction_data = []
    for row in interaction_table.find_all('tr')[1:]:
        columns = row.find_all('td')
        if len(columns) == 3:
            component1 = columns[0].get_text(strip=True)
            component2 = columns[1].get_text(strip=True)
            content = columns[2].get_text(strip=True)
            
            interaction_data.append({
                '성분1': component1,
                '성분2': component2,
                '내용': content
            })

    db_df = pd.DataFrame(interaction_data)
    return db_df

# 스크래핑할 URL
target_url = "https://www.health.kr/searchDrug/result_interaction.asp?drug_cd=2021082400002"

# 함수를 실행하여 데이터를 가져옵니다.
drug_interaction_db = scrape_drug_interaction_info(target_url)

# 결과 출력
if drug_interaction_db is not None:
    print("✅ 약물 상호작용 정보 스크래핑 성공!")
    print(drug_interaction_db)