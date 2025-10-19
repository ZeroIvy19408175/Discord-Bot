import logging
import os
import google.generativeai as genai
from typing import List, Dict, TypedDict, Optional
import chromadb
import uuid
from dotenv import load_dotenv

from ..tokenizer import Tokenizer, count_tokens


# --- 데이터 구조 정의 (Data Structures) ---
class HypaV3Settings(TypedDict):
    summarization_model: str;
    embedding_model: str;
    summarization_prompt: str
    memory_tokens_ratio: float;
    max_chats_per_summary: int
    recent_memory_ratio: float;
    similar_memory_ratio: float


class OpenAIChat(TypedDict):
    role: str;
    content: str;
    memo: Optional[str]


class Chat(TypedDict):
    # ChromaDB를 사용하므로 이 부분은 사실상 비어있게 됩니다.
    hypaV3Data: Optional[Dict]


# --- ChromaDB 클라이언트 설정 ---
db_path = "risu_memory_db"  # DB 파일이 저장될 폴더 이름
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(name="summaries")
logging.info(f"ChromaDB collection '{collection.name}' loaded with {collection.count()} entries from '{db_path}'.")

load_dotenv() # .env 파일 로드

# --- Gemini API 설정 ---
if api_key := os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY environment variable not set.")
tokenizer = Tokenizer()


# --- 헬퍼 함수 (Helper Functions) ---
def get_embedding(text: str, model: str = "text-embedding-004") -> List[float]:
    text = text.replace("\n", " ")
    result = genai.embed_content(model=model, content=text)
    return result['embedding']


async def summarize_for_hypa(text_to_summarize: str, settings: HypaV3Settings) -> str:
    prompt = settings['summarization_prompt']
    full_prompt = f"{text_to_summarize}\n\n{prompt}\n\nOutput:"
    try:
        model = genai.GenerativeModel(settings['summarization_model'])
        response = await model.generate_content_async(full_prompt)
        return response.text.strip() or text_to_summarize
    except Exception as e:
        print(f"Error during Hypa summarization: {e}")
        return text_to_summarize


# --- 핵심 로직: HypaMemory v3 (ChromaDB 버전) ---
async def hypa_memory_v3(
        chats: List[OpenAIChat], current_tokens: int, max_context_tokens: int,
        room: Chat, settings: HypaV3Settings,
) -> dict:
    log_prefix = "[HypaV3-Chroma]"
    memory_prompt_tag = "Past Events Summary"
    start_idx = 0  # 요약을 시작할 채팅 인덱스

    # 1. 요약 단계 (Summarization Phase)
    if current_tokens > max_context_tokens:
        print(f"{log_prefix} Context limit exceeded. Starting summarization process.")
        to_summarize_batch, tokens_to_be_removed = [], 0

        for i in range(start_idx, len(chats) - 3):  # 마지막 3개 메시지는 요약에서 제외
            chat = chats[i]
            if len(to_summarize_batch) < settings['max_chats_per_summary']:
                if chat.get('memo') == 'NewChat' or not chat.get('content', '').strip(): continue
                to_summarize_batch.append(chat)
                tokens_to_be_removed += tokenizer.count_chat_tokens(chat)
            else:
                break

        if to_summarize_batch:
            stringlized_chat = "\n".join([f"{c['role']}: {c['content']}" for c in to_summarize_batch])
            summary_text = await summarize_for_hypa(stringlized_chat, settings)
            summary_embedding = get_embedding(summary_text, model=settings['embedding_model'])
            summary_id = str(uuid.uuid4())

            collection.add(ids=[summary_id], embeddings=[summary_embedding], metadatas=[{"text": summary_text}])

            current_tokens -= tokens_to_be_removed
            start_idx += len(to_summarize_batch)
            print(f"{log_prefix} New summary saved to ChromaDB. Total summaries: {collection.count()}.")

    # 2. 기억 선택 단계 (Memory Selection Phase)
    memory_content = ""
    if collection.count() > 0:
        available_memory_tokens = max_context_tokens * settings['memory_tokens_ratio']
        recent_chats_for_query = [c for c in chats[-3:] if c.get('content', '').strip()]

        if recent_chats_for_query:
            query_text = "\n".join([c['content'] for c in recent_chats_for_query])
            query_embedding = get_embedding(query_text, model=settings['embedding_model'])

            # ChromaDB에 현재 대화와 가장 유사한 요약문 5개를 쿼리
            results = collection.query(query_embeddings=[query_embedding], n_results=5)

            similar_summaries = results['metadatas'][0] if results['metadatas'] else []

            selected_texts, consumed_tokens = [], 0
            for summary_meta in similar_summaries:
                summary_text = summary_meta['text']
                summary_tokens = count_tokens(summary_text)
                if consumed_tokens + summary_tokens <= available_memory_tokens:
                    selected_texts.append(summary_text)
                    consumed_tokens += summary_tokens
            memory_content = "\n\n".join(selected_texts)

    # 3. 최종 조립 단계 (Final Assembly Phase)
    final_memory_prompt = f"<{memory_prompt_tag}>\n{memory_content}\n</{memory_prompt_tag}>" if memory_content else ""
    final_memory_tokens = count_tokens(final_memory_prompt)

    final_chats = chats[start_idx:]
    final_tokens = sum(tokenizer.count_chat_tokens(c) for c in final_chats) + final_memory_tokens

    while final_tokens > max_context_tokens and len(final_chats) > 1:
        removed_chat = final_chats.pop(0)
        final_tokens -= tokenizer.count_chat_tokens(removed_chat)

    if final_memory_prompt:
        final_chats.insert(0, {"role": "system", "content": final_memory_prompt, "memo": "hypaMemory"})

    print(f"{log_prefix} Final context ready. Tokens: {final_tokens}. Chats: {len(final_chats)}.")

    # 이제 memory_data를 반환할 필요가 없습니다.
    return {"current_tokens": final_tokens, "chats": final_chats, "error": None}