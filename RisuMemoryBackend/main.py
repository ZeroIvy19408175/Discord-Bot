import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

# 수정된 임포트 경로
from risu_memory_backend.tokenizer import Tokenizer, count_chat_history_tokens
from risu_memory_backend.memory.supa_memory import supa_memory, OpenAIChat as SupaOpenAIChat, Chat as SupaChat, \
    Character as SupaCharacter
from risu_memory_backend.memory.hypa_memory import hypa_memory_v3, HypaV3Settings, OpenAIChat as HypaOpenAIChat, \
    Chat as HypaChat

# --- FastAPI App Initialization ---
app = FastAPI(
    title="RisuAI Long-Term Memory Backend (ChromaDB Edition)",
    description="A Python implementation of RisuAI's long-term memory systems using ChromaDB and Google Gemini API.",
    version="2.0.0",
)

tokenizer = Tokenizer()


# --- Pydantic Models for API ---
class ChatMessage(BaseModel):
    role: str
    content: str
    memo: Optional[str] = None


class ProcessChatRequest(BaseModel):
    messages: List[ChatMessage]
    memory_type: Literal['supa', 'hypa'] = Field("hypa", description="The type of memory system to use.")
    max_context_tokens: int = Field(8192, description="The maximum token limit for the context.")
    character_name: str = Field("Risu", description="The name of the character.")
    hypa_settings: Optional[HypaV3Settings] = None
    # room_data는 이제 HypaMemory에서 사용되지 않지만, SupaMemory와의 호환성을 위해 남겨둡니다.
    room_data: Dict = Field({}, description="Persistent data for the chat room, used by SupaMemory.")


# --- API Endpoint ---
@app.post("/process_chat/")
async def process_chat(request: ProcessChatRequest):
    current_tokens = count_chat_history_tokens([msg.dict() for msg in request.messages])

    if current_tokens <= request.max_context_tokens:
        return {
            "processed_messages": request.messages,
            "final_tokens": current_tokens,
            # updated_room_data는 이제 별 의미가 없지만, 봇과의 호환성을 위해 빈 객체를 보냅니다.
            "updated_room_data": {},
            "info": "Context window not exceeded, no memory processing needed."
        }

    if request.memory_type == 'supa':
        # SupaMemory는 여전히 room_data를 사용합니다 (휘발성).
        supa_chats: List[SupaOpenAIChat] = [msg.dict() for msg in request.messages]
        supa_room: SupaChat = {"supaMemoryData": request.room_data.get("supaMemoryData")}
        supa_char: SupaCharacter = {"name": request.character_name}
        result = await supa_memory(
            chats=supa_chats, current_tokens=current_tokens, max_context_tokens=request.max_context_tokens,
            room=supa_room, char=supa_char
        )
        if result.get("error"): raise HTTPException(status_code=500, detail=result["error"])
        # SupaMemory는 여전히 room_data를 업데이트합니다.
        updated_data = {"supaMemoryData": result.get("memory")}
        return {
            "processed_messages": result["chats"], "final_tokens": result["current_tokens"],
            "updated_room_data": updated_data,
            "info": f"SupaMemory processed. Last summarized message ID: {result.get('last_id')}"
        }

    elif request.memory_type == 'hypa':
        hypa_chats: List[HypaOpenAIChat] = [msg.dict() for msg in request.messages]
        # ChromaDB를 사용하므로 room_data를 전달할 필요가 없습니다.
        hypa_room: HypaChat = {}
        hypa_settings = request.hypa_settings or HypaV3Settings(
            summarization_model='gemini-flash-latest', embedding_model='text-embedding-004',
            summarization_prompt='[Summarize the ongoing role story, focusing on key events, character progression, and unresolved plot points.]',
            memory_tokens_ratio=0.25, max_chats_per_summary=8,
            recent_memory_ratio=0.3, similar_memory_ratio=0.5,
        )
        result = await hypa_memory_v3(
            chats=hypa_chats, current_tokens=current_tokens, max_context_tokens=request.max_context_tokens,
            room=hypa_room, settings=hypa_settings
        )
        if result.get("error"): raise HTTPException(status_code=500, detail=result["error"])

        # HypaMemory는 더 이상 room_data를 반환하지 않습니다.
        return {
            "processed_messages": result["chats"], "final_tokens": result["current_tokens"],
            "updated_room_data": {},  # 빈 객체 반환
            "info": f"HypaMemory (ChromaDB) processed."
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid memory_type specified.")


@app.get("/")
async def root():
    return {"message": "RisuAI Long-Term Memory Backend (ChromaDB Edition) is running."}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)