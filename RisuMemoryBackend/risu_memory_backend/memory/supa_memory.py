import os
import google.generativeai as genai
from typing import List, Dict, TypedDict, Optional

# 상위 폴더의 tokenizer를 임포트하기 위해 경로를 수정합니다.
from ..tokenizer import Tokenizer, count_tokens


# --- 데이터 구조 정의 (Data Structures) ---
class OpenAIChat(TypedDict):
    role: str
    content: str
    memo: Optional[str]


class Chat(TypedDict):
    # 이 컨텍스트에서는 단순화된 버전으로 사용합니다.
    supaMemoryData: Optional[str]


class Character(TypedDict):
    name: str


# --- Gemini API 설정 ---
# .env 파일에서 키를 읽어옵니다.
if api_key := os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY environment variable not set.")


# --- 핵심 함수 (Core Functions) ---
async def summarize(text_to_summarize: str, supa_model_type: str = 'gemini-flash-latest',
                    supa_memory_prompt: str = "") -> str:
    """
    주어진 텍스트를 Gemini API를 사용하여 요약합니다.
    """
    if not supa_memory_prompt:
        supa_memory_prompt = "[Summarize the ongoing role story, It must also remove redundancy and unnecessary text and content from the output to reduce tokens for gpt3 and other sublanguage models]"

    prompt_body = f"{text_to_summarize}\n\n{supa_memory_prompt}\n\nOutput:"

    try:
        model = genai.GenerativeModel(supa_model_type)
        response = await model.generate_content_async(prompt_body)

        result = response.text.strip()
        if not result:
            raise ValueError("Summarization returned an empty result.")
        return result
    except Exception as e:
        print(f"Error during summarization: {e}")
        # 실패 시 원본 텍스트를 반환하여 데이터 손실을 방지합니다.
        return text_to_summarize


async def supa_memory(
        chats: List[OpenAIChat],
        current_tokens: int,
        max_context_tokens: int,
        room: Chat,
        char: Character,
) -> dict:
    """
    컨텍스트 창이 초과되면 대화를 요약하여 장기 기억을 관리합니다.
    RisuAI의 supaMemory.ts 로직을 Python으로 단순화하여 번역한 버전입니다.
    """
    if current_tokens <= max_context_tokens:
        return {
            "current_tokens": current_tokens,
            "chats": chats,
            "error": None
        }

    print(
        f"Context limit exceeded. Current tokens: {current_tokens}. Max tokens: {max_context_tokens}. Starting summarization.")

    supa_memory_summary = ''
    last_id = ''

    # 기존 요약 내용이 있으면 불러옵니다.
    if room.get('supaMemoryData') and len(room['supaMemoryData']) > 4:
        supa_memory_summary = room['supaMemoryData']
        current_tokens += count_tokens(supa_memory_summary)

    # 컨텍스트가 맞을 때까지 요약을 반복합니다.
    while current_tokens > max_context_tokens:
        chunk_size = 0
        stringlized_chat = ''
        splice_len = 0  # 대화 기록의 시작 부분에서 제거할 메시지 수
        max_chunk_tokens = max_context_tokens / 3  # 컨텍스트의 약 1/3을 요약

        for i, chat_message in enumerate(chats):
            # 시스템 메시지는 요약 대상에서 제외
            if chat_message['role'] == 'system':
                continue

            message_tokens = count_chat_tokens(chat_message)
            if chunk_size + message_tokens > max_chunk_tokens and stringlized_chat:
                last_id = chat_message.get('memo', '')
                break

            # 대화 내용을 "user: ..." 또는 "assistant: ..." 형식의 문자열로 만듭니다.
            stringlized_chat += f"{char['name'] if chat_message['role'] == 'assistant' else 'user'}: {chat_message['content']}\n\n"
            splice_len = i + 1
            current_tokens -= message_tokens
            chunk_size += message_tokens

        if not stringlized_chat:
            # 요약할 청크를 만들지 못한 경우
            return {
                "current_tokens": current_tokens,
                "chats": chats,
                "error": "Not enough tokens to summarize or failed to create a summarization chunk."
            }

        # 요약된 부분을 대화 기록에서 제거합니다.
        chats = chats[splice_len:]

        # 새로운 요약 부분을 생성합니다.
        new_summary_part = await summarize(stringlized_chat)
        new_summary_tokens = count_tokens(new_summary_part)

        # 전체 요약문에 새로운 요약 부분을 추가합니다.
        supa_memory_summary = f"{supa_memory_summary}\n\n{new_summary_part}".strip()
        current_tokens += new_summary_tokens

    # 최종 요약문을 시스템 메시지로 대화 기록 맨 앞에 추가합니다.
    chats.insert(0, {
        "role": "system",
        "content": supa_memory_summary,
        "memo": "supaMemory"
    })

    return {
        "current_tokens": current_tokens,
        "chats": chats,
        "memory": supa_memory_summary,  # 저장할 전체 요약 데이터
        "last_id": last_id,
        "error": None
    }