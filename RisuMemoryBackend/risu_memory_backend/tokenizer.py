import tiktoken
from typing import List, Dict

class Tokenizer:
    def __init__(self, model_name: str = "gpt-4"):
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def count_chat_tokens(self, message: Dict[str, str]) -> int:
        """
        Counts the number of tokens in a single chat message.
        """
        num_tokens = 4
        for key, value in message.items():
            if value:
                num_tokens += self.count_tokens(value)
            if key == "name":
                num_tokens -= 1
        return num_tokens

    def count_chat_history_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Counts the number of tokens in a list of chat messages.
        """
        num_tokens = 0
        for message in messages:
            num_tokens += self.count_chat_tokens(message)
        num_tokens += 3
        return num_tokens

# Global tokenizer instance for easy import
tokenizer = Tokenizer()

def count_tokens(text: str) -> int:
    return tokenizer.count_tokens(text)

def count_chat_tokens(message: Dict[str, str]) -> int:
    return tokenizer.count_chat_tokens(message)

def count_chat_history_tokens(messages: List[Dict[str, str]]) -> int:
    return tokenizer.count_chat_history_tokens(messages)