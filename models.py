import ollama


class OllamaModel:
    def __init__(self, model: str, host: str) -> None:
        self._model = model
        self._client = ollama.Client(host=host)

    def chat(self, messages: list[dict]) -> str:
        try:
            response = self._client.chat(model=self._model, messages=messages)
            return response.message.content
        except Exception as e:
            print(f"OllamaModel error: {e}")
            raise
