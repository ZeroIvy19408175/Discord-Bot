# json_to_bot_memory.py

import json
import os
import uuid


def convert_ai_studio_to_bot_format(json_file_path: str, user_name: str = "default_user"):
    """
    Google AI Studio의 JSON을 디스코드 봇의 user_chat_histories 형식으로 변환합니다.
    """
    if not os.path.exists(json_file_path):
        print(f"오류: 파일을 찾을 수 없습니다 - {json_file_path}")
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        bot_history = []
        if 'chunkedPrompt' in data and 'chunks' in data['chunkedPrompt']:
            for chunk in data['chunkedPrompt']['chunks']:
                role = chunk.get('role', 'unknown')
                text = chunk.get('text', '').strip()
                is_thought = chunk.get('isThought', False)

                if not text or is_thought:
                    continue

                # 디스코드 봇의 role 형식(user/assistant)에 맞게 변환
                bot_role = "assistant" if role == "model" else "user"

                bot_history.append({
                    "role": bot_role,
                    "content": text,
                    "memo": str(uuid.uuid4())  # 고유한 memo 생성
                })

        output_data = {
            user_name: bot_history
        }

        output_filename = "bot_short_term_memory_imported.json"
        with open(output_filename, 'w', encoding='utf-8') as f_out:
            json.dump(output_data, f_out, ensure_ascii=False, indent=4)

        print(f"성공: '{json_file_path}' 파일을 봇 메모리 형식인 '{output_filename}'으로 변환했습니다.")
        print(f"이 파일을 'bot_short_term_memory.json'으로 이름을 바꾸고 봇을 실행하면 대화 기록이 로드됩니다.")

    except Exception as e:
        print(f"변환 중 오류 발생: {e}")


if __name__ == '__main__':
    input_file = "your_chat_history.json"
    # 이 대화 기록을 어떤 유저의 기록으로 저장할지 이름을 지정합니다.
    target_user = "ZeroIvy19408175"
    convert_ai_studio_to_bot_format(input_file, user_name=target_user)