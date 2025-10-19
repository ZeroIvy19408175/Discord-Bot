# bot/gemini_api.py
import google.generativeai as genai
import google.ai.generativelanguage as glm
import base64
from config import MODEL_NAME, SYSTEM_INSTRUCTION, GEMINI_API_KEY
from utils import multiply, get_weather, get_uptime
import logging

# ----- Logging Setup -----
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- Gemini Model Configuration -----
generation_config = {
    "temperature": 0,  # Set temperature to 0 for more deterministic responses
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings_none = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# ----- Gemini Function Declarations -----
multiply_declaration = glm.FunctionDeclaration(
    name="multiply",
    description="두 숫자를 곱합니다.",
    parameters=glm.Schema(
        type=glm.Type.OBJECT,
        properties={
            "a": glm.Schema(type=glm.Type.NUMBER, description="첫 번째 숫자"),
            "b": glm.Schema(type=glm.Type.NUMBER, description="두 번째 숫자"),
        },
        required=["a", "b"],
    ),
)

get_weather_declaration = glm.FunctionDeclaration(
    name="get_weather",
    description="특정 도시의 현재 날씨 정보를 가져옵니다.",
    parameters=glm.Schema(
        type=glm.Type.OBJECT,
        properties={"city": glm.Schema(type=glm.Type.STRING, description="날씨 정보를 가져올 도시 이름")},
        required=["city"],
    ),
)

get_uptime_declaration = glm.FunctionDeclaration(
    name="get_uptime",
    description="봇의 가동 시간을 가져옵니다.",
)


class CustomChatSession:
    def __init__(self, model, history=None):
        logging.debug(f"CustomChatSession __init__: Creating new session.")
        self.chat = model.start_chat(history=None)  # Force new session

    def send_message(self, content: glm.Content) -> glm.Message:
        if not hasattr(content, 'role') or content.role is None:
            logging.debug("send_message: Role is missing. Setting it to 'user'")
            new_content = glm.Content(parts=content.parts, role="user")
        else:
            new_content = content
        logging.debug(f"Sending message with role: {new_content.role}")
        response = self.chat.send_message(new_content)
        return response

    def send_message_with_role(self, content: glm.Content, role: str) -> glm.Message:
        if not hasattr(content, 'role') or content.role is None:
            logging.debug(f"send_message_with_role: Role is missing. Setting it to {role}")
            new_content = glm.Content(parts=content.parts, role=role)
        else:
            new_content = content
        logging.debug(f"Sending message with role: {new_content.role}")
        response = self.chat.send_message(new_content)
        return response

    def get_history(self) -> list[glm.Content]:
        return [message for message in self.chat.history]

    def summarize(self, content: str) -> str:
        parts = [glm.Part(text=f"이 대화를 요약해줘: {content}")]
        content = glm.Content(parts=parts)
        return self.send_message(content).text


def initialize_gemini_model():
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings_none,
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[multiply_declaration, get_weather_declaration, get_uptime_declaration]
    )
    return model