from chatbot import Conversation


class _MockModel:
    def __init__(self, response: str = "Mock response") -> None:
        self._response = response
        self.last_messages: list[dict] = []

    def chat(self, messages: list[dict]) -> str:
        self.last_messages = messages
        return self._response


class _FailingModel:
    def chat(self, messages: list[dict]) -> str:
        raise RuntimeError("Connection failed")


def test_get_completion_returns_model_response():
    model = _MockModel("Hello!")
    conv = Conversation(model, "You are helpful.")
    assert conv.get_completion() == "Hello!"


def test_get_completion_passes_full_context_to_model():
    model = _MockModel("Hi")
    conv = Conversation(model, "You are helpful.")
    conv.collect_messages("user", "What is tea?")
    conv.get_completion()

    roles = [m["role"] for m in model.last_messages]
    contents = [m["content"] for m in model.last_messages]
    assert roles[0] == "system"
    assert "What is tea?" in contents


def test_collect_messages_appends_to_context():
    model = _MockModel()
    conv = Conversation(model, "System prompt")
    conv.collect_messages("user", "Hello")
    conv.collect_messages("assistant", "Hi")
    conv.get_completion()
    # context = system + user + assistant = 3 messages passed to model
    assert len(model.last_messages) == 3


def test_get_completion_returns_fallback_on_error():
    conv = Conversation(_FailingModel(), "System prompt")
    result = conv.get_completion()
    assert result == "Sorry, I encountered an error."
