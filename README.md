
# 🧠 RisuAI Long-Term Memory Backend (ChromaDB + Gemini Edition)

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![VectorDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-3377FF)](https://www.trychroma.com/)
[![Language Model](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4)](https://ai.google.dev/gemini-api)

## 🌟 프로젝트 소개

이 프로젝트는 **RisuAI**의 혁신적인 **장기기억(Long-Term Memory)** 시스템인 **HypaMemory V3** 로직을 Google Gemini API와 ChromaDB를 사용하여 Python으로 재구현한 백엔드 서버입니다.

LLM의 단기 기억(컨텍스트 창) 한계를 극복하고, 수많은 대화 속에서도 **맥락과 관련성이 가장 높은 과거 기억**을 지능적으로 검색하여 LLM에 전달하는 역할을 합니다.

이 백엔드는 독립적인 FastAPI 서버로 작동하며, 디스코드 봇이나 다른 채팅 애플리케이션에 플러그인처럼 쉽게 연결하여 LLM에 영구적인 기억 능력을 부여할 수 있습니다.

### 핵심 기능

*   **비휘발성 장기기억:** ChromaDB를 사용하여 임베딩 벡터와 요약 데이터를 디스크에 영구 저장합니다. 봇을 껐다 켜도 기억이 초기화되지 않습니다.
*   **지능적 컨텍스트 관리:** HypaMemory 로직을 통해 현재 대화와 가장 유사한 과거 기억(Vector Search)을 찾아내고, 오래된 대화는 요약하여 컨텍스트 창을 최적화합니다.
*   **API 기반 연동:** `/process_chat/` API 엔드포인트를 통해 외부 클라이언트(디스코드 봇)가 쉽게 접근할 수 있습니다.
*   **Google Gemini Native:** 모든 요약 및 임베딩 작업에 `gemini-flash-latest` 및 `text-embedding-004` 모델을 사용합니다.

## ⚙️ 아키텍처 및 작동 방식

1.  **클라이언트 (Discord Bot):** 대화 기록을 `/process_chat/` 엔드포인트로 전송합니다.
2.  **FastAPI Backend (HypaMemory):**
    *   대화가 길어지면 오래된 부분을 요약하고 임베딩 벡터를 생성합니다.
    *   새 임베딩을 ChromaDB에 저장합니다.
    *   현재 대화와 가장 유사한 과거 기억을 ChromaDB에서 검색합니다.
    *   가장 관련성 높은 과거 기억을 포함한 **최적화된 대화 기록**을 클라이언트에게 반환합니다.
3.  **클라이언트 (Discord Bot):** 백엔드에서 받은 최적화된 기록을 Gemini API에 보내 최종 답변을 생성합니다.

## 🚀 시작하기

### 1. 환경 설정 및 라이브러리 설치

프로젝트 루트에서 터미널을 열고 필요한 라이브러리를 설치합니다.

```bash
# Python 가상 환경 활성화 후 실행 권장
pip install -r requirements.txt
```

### 2. API 키 설정 (필수)

프로젝트 루트 폴더에 `.env` 파일을 생성하고, 당신의 Google Gemini API 키를 입력합니다.

**.env**
```
# RisuMemoryBackend/.env
GEMINI_API_KEY="여기에_당신의_구글_제미나이_API_키를_입력하세요"
```

### 3. 서버 실행

터미널에서 서버를 실행합니다.

```bash
# --reload 옵션은 코드 변경 시 자동 재시작을 지원합니다.
uvicorn main:app --reload
```

서버가 성공적으로 실행되면 다음과 같은 메시지를 확인할 수 있습니다.
`INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`

## 🛠️ 클라이언트 (Discord Bot) 연동

이 백엔드를 사용하는 클라이언트(디스코드 봇)는 `/process_chat/` 엔드포인트로 다음 형식의 POST 요청을 보내야 합니다.

### API 엔드포인트

*   **URL:** `http://127.0.0.1:8000/process_chat/`
*   **Method:** `POST`

### Request Body (JSON)

```json
{
    "messages": [
        {"role": "system", "content": "너는...", "memo": "..." },
        {"role": "user", "content": "안녕", "memo": "..." }
    ],
    "memory_type": "hypa",
    "max_context_tokens": 8192,
    "character_name": "Risu",
    "room_data": {}
}
```

### Response Body (JSON)

```json
{
    "processed_messages": [
        {"role": "system", "content": "<Past Events Summary>...</Past Events Summary>", "memo": "hypaMemory"},
        {"role": "user", "content": "안녕", "memo": "..." }
    ],
    "final_tokens": 78,
    "updated_room_data": {},
    "info": "HypaMemory (ChromaDB) processed."
}
```
**참고:** 클라이언트(디스코드 봇)는 응답으로 받은 `processed_messages`를 LLM에 보내 답변을 생성하면 됩니다.
