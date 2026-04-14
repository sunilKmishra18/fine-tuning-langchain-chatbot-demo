from typing import Protocol


class ChatModel(Protocol):
    def chat(self, messages: list[dict]) -> str: ...
