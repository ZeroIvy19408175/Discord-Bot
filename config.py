# bot/config.py
import pytz
from datetime import timedelta
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
personas_path = os.path.join(grandparent_dir, "personas", "personas")
sys.path.append(os.path.dirname(personas_path))
from personas import *
from dotenv import load_dotenv

load_dotenv()

# config.py 파일에 추가

# ----- 키워드 설정 (JEBI_KEYWORDS) -----
# 봇이 응답해야 하는 키워드를 여기에 정의합니다.
JEBI_KEYWORDS = {"제비야", "제비", "야제비", "제비님", "제비봇", "야제비야"}


# ----- Constants -----
MODEL_NAME = "gemini-flash-latest"
TIMEZONE = pytz.timezone('Asia/Shanghai')
INACTIVE_SESSION_TIMEOUT = timedelta(minutes=5)
DATA_DIR = "bot_data"
SYSTEM_INSTRUCTION = standard_inst_v2
# ----- API Configurations -----




# ----- API 설정 (API Configurations) -----
# 각 서비스에서 발급받은 API 키를 따옴표 안에 입력하세요.

# 1. Discord Bot Token
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# 2. Google Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 3. OpenWeather API Key
OPENWEATHER_API = os.getenv("OPENWEATHER_API")

# 4. SerpApi API Key
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# ----- 메모리 백엔드 서버 주소 -----
# 로컬에서 실행 중인 FastAPI 서버의 주소입니다.
MEMORY_API_URL = "http://127.0.0.1:8000/process_chat/"

# ----- 모든 키가 제대로 로드되었는지 확인 (선택 사항) -----
if not all([DISCORD_BOT_TOKEN, GEMINI_API_KEY, OPENWEATHER_API, SERPAPI_API_KEY]):
    print("경고: .env 파일에 필요한 API 키 중 일부가 설정되지 않았습니다.")