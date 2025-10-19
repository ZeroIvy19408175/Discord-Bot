# bot/utils.py
import requests
from datetime import datetime, timezone
from serpapi import GoogleSearch
import os
from typing import Optional


def multiply(a: float, b: float) -> float:
    """Returns the product of two numbers."""
    return a * b


def get_weather(city: str, api_key: str) -> str:
    """Gets the weather information for a given city."""
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
        "lang": "kr"
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        weather_description = weather_data["weather"][0]["description"]
        temperature = weather_data["main"]["temp"]
        return f"{city}의 현재 날씨는 {weather_description}이며, 기온은 {temperature}°C입니다."
    except Exception as e:
        return f"날씨 정보를 가져오는 데 실패했습니다: {e}"


def get_uptime(start_time: Optional[datetime] = None) -> str:
    """Returns the bot's uptime."""
    if start_time is None:
        return "봇 시작 시간을 알 수 없습니다."
    # UTC 기준 시간대를 가정하고 계산합니다.
    uptime = datetime.now(timezone.utc) - start_time
    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}시간 {minutes}분 {seconds}초"


def search_web(query: str) -> str:
    """
    Performs a Google search for the given query and returns the top results.
    Requires the SERPAPI_API_KEY to be set in the .env file.
    """
    try:
        # .env 파일에서 키를 로드합니다.
        from config import SERPAPI_API_KEY
        if not SERPAPI_API_KEY:
            return "웹 검색 API 키(SERPAPI_API_KEY)가 설정되지 않았습니다."

        params = {
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "engine": "google",
            "gl": "kr",
            "hl": "ko",
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        # 검색 결과 파싱
        answer_box = results.get("answer_box")
        organic_results = results.get("organic_results", [])

        if answer_box:
            snippet = answer_box.get("snippet") or answer_box.get("answer")
            if snippet:
                return f"검색 결과 (직접 답변): {snippet}"

        if organic_results:
            top_results_summary = "웹 검색 결과 요약:\n"
            for i, res in enumerate(organic_results[:3]):
                title = res.get("title", "제목 없음")
                snippet = res.get("snippet", "내용 없음")
                # 링크는 너무 길어질 수 있으므로 제거하고 제목과 내용만 요약합니다.
                top_results_summary += f"{i + 1}. {title}: {snippet}\n"
            return top_results_summary

        return "웹에서 관련 정보를 찾지 못했습니다."

    except Exception as e:
        # 오류가 나면 오류 메시지를 LLM에게 보내 봇이 사용자에게 자연스럽게 설명하도록 합니다.
        return f"웹 검색 중 내부 오류가 발생했습니다: {e}"