from protocols import ChatModel


class Conversation:
    def __init__(self, model: ChatModel, system_prompt: str) -> None:
        self._model = model
        self._context: list[dict] = [{"role": "system", "content": system_prompt}]

    def collect_messages(self, role: str, message: str) -> None:
        self._context.append({"role": role, "content": message})

    def get_completion(self) -> str:
        try:
            response = self._model.chat(self._context)
            print(f"\n Assistant: {response}\n")
            return response
        except Exception as e:
            print(f"Error getting completion: {e}")
            return "Sorry, I encountered an error."
