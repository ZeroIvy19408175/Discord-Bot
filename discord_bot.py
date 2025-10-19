import discord
from discord.ext import commands, tasks
import os
import logging
import json
import uuid
import base64
from datetime import datetime, timezone
import httpx
import google.generativeai as genai
import google.ai.generativelanguage as glm

# config.py에서 모든 설정을 가져옵니다.
from config import (
    DISCORD_BOT_TOKEN, MEMORY_API_URL, GEMINI_API_KEY,
    SYSTEM_INSTRUCTION, OPENWEATHER_API, SERPAPI_API_KEY
)
# utils.py에서 실제 실행할 함수들을 가져옵니다.
from utils import get_uptime, get_weather, search_web

# ----- 기본 설정 -----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='제', intents=intents)
start_time = datetime.now(timezone.utc)

# ----- Tools (함수) 정의 -----
tools = [
    glm.Tool(function_declarations=[
        glm.FunctionDeclaration(
            name="get_weather",
            description="특정 도시의 현재 날씨 정보를 가져옵니다.",
            parameters=glm.Schema(type=glm.Type.OBJECT, properties={
                "city": glm.Schema(type=glm.Type.STRING, description="날씨 정보를 가져올 도시 이름 (영어로)")}, required=["city"]),
        ),
        glm.FunctionDeclaration(
            name="get_uptime",
            description="봇의 현재 가동 시간을 알려줍니다.",
        ),
        glm.FunctionDeclaration(
            name="search_web",
            description="최신 정보나 특정 주제에 대해 웹에서 검색합니다.",
            parameters=glm.Schema(type=glm.Type.OBJECT,
                                  properties={"query": glm.Schema(type=glm.Type.STRING, description="검색할 내용")},
                                  required=["query"]),
        ),
    ])
]
available_functions = {
    "get_weather": get_weather,
    "get_uptime": get_uptime,
    "search_web": search_web,
}

# ----- Google Gemini API 클라이언트 설정 -----
generation_model = None
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        generation_model = genai.GenerativeModel(
            'gemini-flash-latest',
            system_instruction=SYSTEM_INSTRUCTION,
            tools=tools
        )
        logging.info("Gemini API 클라이언트가 페르소나 및 모든 도구와 함께 설정되었습니다.")
    else:
        logging.warning("경고: config.py에 GEMINI_API_KEY가 설정되지 않았습니다.")
except Exception as e:
    logging.error(f"Gemini API 설정 중 오류 발생: {e}")

# ----- 비휘발성 단기기억 관리 -----
MEMORY_FILE_PATH = "bot_short_term_memory.json"
user_chat_histories = {}  # 유저별 대화 기록 (이미지 포함 단기기억)


def save_memory_to_disk():
    """현재 단기기억(대화 기록)을 JSON 파일에 저장합니다."""
    try:
        with open(MEMORY_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(user_chat_histories, f, ensure_ascii=False, indent=4)
        logging.info(f"단기 기억을 '{MEMORY_FILE_PATH}'에 저장했습니다.")
    except Exception as e:
        logging.error(f"단기 기억 저장 중 오류: {e}")


def load_memory_from_disk():
    """JSON 파일에서 단기기억(대화 기록)을 불러옵니다."""
    global user_chat_histories
    try:
        if os.path.exists(MEMORY_FILE_PATH):
            with open(MEMORY_FILE_PATH, "r", encoding="utf-8") as f:
                user_chat_histories = json.load(f)
            logging.info(f"'{MEMORY_FILE_PATH}'에서 단기 기억을 불러왔습니다.")
    except Exception as e:
        logging.error(f"단기 기억 로딩 중 오류: {e}")


@tasks.loop(minutes=5)
async def periodic_save_task():
    save_memory_to_disk()


# ----- Discord 이벤트 핸들러 -----

@bot.event
async def on_ready():
    load_memory_from_disk()
    periodic_save_task.start()
    logging.info(f'{bot.user.name} 온라인! 모든 기억이 로드되었습니다.')


@bot.event
async def on_close():
    logging.info("봇이 종료됩니다. 마지막으로 기억을 저장합니다...")
    save_memory_to_disk()


@bot.event
async def on_message(message):
    if message.author == bot.user or message.author.bot: return
    if not message.content.strip() and not message.attachments: return
    if not message.content.startswith(bot.command_prefix):
        await process_chat_message(message)
    await bot.process_commands(message)


# ----- 핵심 대화 처리 로직 -----

async def process_chat_message(message):
    user_name = message.author.name
    message_memo = str(uuid.uuid4())
    user_message_record = {"role": "user", "content": message.content, "memo": message_memo}

    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                try:
                    image_bytes = await attachment.read()
                    user_message_record["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
                    user_message_record["mime_type"] = attachment.content_type
                    logging.info(f"이미지 첨부파일을 단기 기억에 저장: {attachment.filename}")
                    break
                except Exception as e:
                    await message.channel.send("이미지를 처리하는 데 실패했어.");
                    return

    if user_name not in user_chat_histories: user_chat_histories[user_name] = []
    user_chat_histories[user_name].append(user_message_record)

    text_only_history = [{"role": msg["role"], "content": msg["content"], "memo": msg.get("memo")} for msg in
                         user_chat_histories[user_name]]
    payload = {
        "messages": text_only_history, "memory_type": "hypa", "max_context_tokens": 8192,
        "character_name": bot.user.name, "room_data": {}
    }

    async with message.channel.typing():
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(MEMORY_API_URL, json=payload)
                response.raise_for_status()
            memory_response = response.json()
            processed_text_messages = memory_response["processed_messages"]

            final_gemini_messages = []
            for processed_msg in processed_text_messages:
                memo, gemini_parts = processed_msg.get("memo"), []
                if processed_msg.get("content"): gemini_parts.append(glm.Part(text=processed_msg["content"]))
                if memo:
                    for original_msg in user_chat_histories[user_name]:
                        if original_msg.get("memo") == memo and "image_base64" in original_msg:
                            gemini_parts.append(glm.Part(inline_data=glm.Blob(mime_type=original_msg["mime_type"],
                                                                              data=base64.b64decode(
                                                                                  original_msg["image_base64"]))))
                            break
                if gemini_parts:
                    final_gemini_messages.append(
                        {"role": "model" if processed_msg["role"] == "assistant" else "user", "parts": gemini_parts})

            if not generation_model:
                await message.channel.send("모델이 준비되지 않았어.");
                return

            chat_session = generation_model.start_chat(history=final_gemini_messages[:-1])
            final_user_message_for_gemini = final_gemini_messages[-1]['parts'] if final_gemini_messages else []
            if not final_user_message_for_gemini: return
            llm_response = await chat_session.send_message_async(final_user_message_for_gemini)

            if not llm_response.candidates or not llm_response.candidates[0].content.parts:
                logging.info("모델이 응답하지 않기로 결정하여 침묵합니다.")
                user_chat_histories[user_name].append({"role": "assistant", "content": "", "memo": str(uuid.uuid4())})
                return

            while True:
                response_part = llm_response.candidates[0].content.parts[0]
                if response_part.function_call:
                    function_call, fname, args = response_part.function_call, response_part.function_call.name, {k: v
                                                                                                                 for
                                                                                                                 k, v in
                                                                                                                 response_part.function_call.args.items()}
                    logging.info(f"함수 호출: {fname}({args})")
                    if fname in available_functions:
                        if fname == "get_weather": args['api_key'] = OPENWEATHER_API
                        f_response = available_functions[fname](**args)
                        llm_response = await chat_session.send_message_async(glm.Part(
                            function_response=glm.FunctionResponse(name=fname, response={"result": f_response})))
                    else:
                        llm_response = await chat_session.send_message_async(glm.Part(
                            function_response=glm.FunctionResponse(name=fname, response={"result": "알 수 없는 함수"})))
                else:
                    break

            response_text = llm_response.text.strip()
            if response_text:
                await message.channel.send(response_text)

            user_chat_histories[user_name].append(
                {"role": "assistant", "content": response_text, "memo": str(uuid.uuid4())})

        except httpx.RequestError as e:
            await message.channel.send(f"메모리 서버 연결 실패. 🧠 (에러: {e})")
        except Exception as e:
            await message.channel.send(f"처리 중 오류 발생. 🤯 (에러: {e})")
            logging.error(f"처리 중 오류 발생: {e}", exc_info=True)
        finally:
            save_memory_to_disk()


# ----- Discord 커맨드 -----

@bot.command()
async def 업타임(ctx):
    await ctx.send(f"내가 깨어난 지: {get_uptime(start_time)}")


@bot.command()
async def 기억초기화(ctx):
    user_name = ctx.author.name
    if user_name in user_chat_histories: del user_chat_histories[user_name]
    save_memory_to_disk()  # 초기화된 상태를 바로 저장
    await ctx.send(f"{user_name}와의 단기 기억을 모두 지웠어. (장기기억은 백엔드 서버에서 별도로 관리돼!)")


if __name__ == "__main__":
    if not all([DISCORD_BOT_TOKEN, MEMORY_API_URL, GEMINI_API_KEY, SERPAPI_API_KEY]):
        raise ValueError("필수 API 키가 config.py에 없습니다.")
    try:
        bot.run(DISCORD_BOT_TOKEN)
    except KeyboardInterrupt:
        print("\n키보드 인터럽트 감지. 종료 전 기억을 저장합니다...")
        save_memory_to_disk()
    finally:
        print("봇 프로그램을 종료합니다.")