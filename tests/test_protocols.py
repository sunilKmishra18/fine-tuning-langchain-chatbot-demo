from protocols import ChatModel


def test_chat_model_protocol_satisfied_by_duck_type():
    class FakeModel:
        def chat(self, messages: list[dict]) -> str:
            return "ok"

    model: ChatModel = FakeModel()
    assert model.chat([]) == "ok"
