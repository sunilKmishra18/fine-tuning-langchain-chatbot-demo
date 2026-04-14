import pytest
from unittest.mock import MagicMock, patch
from models import OllamaModel


def test_chat_returns_message_content():
    mock_response = MagicMock()
    mock_response.message.content = "Test response"

    with patch("models.ollama.Client") as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        model = OllamaModel(model="llama3.2", host="http://localhost:11434")
        result = model.chat([{"role": "user", "content": "Hello"}])

    assert result == "Test response"


def test_chat_calls_client_with_correct_model_and_messages():
    mock_response = MagicMock()
    mock_response.message.content = "Response"

    with patch("models.ollama.Client") as MockClient:
        mock_client = MockClient.return_value
        mock_client.chat.return_value = mock_response
        model = OllamaModel(model="llama3.2", host="http://localhost:11434")
        messages = [{"role": "user", "content": "Hello"}]
        model.chat(messages)

    mock_client.chat.assert_called_once_with(model="llama3.2", messages=messages)


def test_chat_re_raises_on_exception():
    with patch("models.ollama.Client") as MockClient:
        MockClient.return_value.chat.side_effect = RuntimeError("Connection refused")
        model = OllamaModel(model="llama3.2", host="http://localhost:11434")

        with pytest.raises(RuntimeError, match="Connection refused"):
            model.chat([{"role": "user", "content": "Hello"}])
