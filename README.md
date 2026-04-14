# SOLID Refactor Design — TeaCrafters Chatbot

**Scope:** TeaCrafters Chatbot demonstration with applying all five SOLID principles.

---

## 1. Goal

Apply five focused modules. Each module has one reason to change. The LLM provider is abstracted behind a `ChatModel` protocol so future providers can be swapped in without touching chat or dataset logic.

---

## 2. Module Structure

```
fine-tuning-langchain-chat/
├── protocols.py       # ChatModel Protocol — abstraction boundary
├── models.py          # OllamaModel — concrete ChatModel implementation
├── dataset.py         # DatasetLoader, DatasetValidator, TokenCounter
├── chatbot.py         # Conversation — owns context, depends on ChatModel
└── main.py            # Entry point — wires everything together
```

The existing `app.py` is replaced by this structure. No other files change.

---

## 3. Classes & Interfaces

### `protocols.py`
```python
class ChatModel(Protocol):
    def chat(self, messages: list[dict]) -> str: ...
```
- No project imports. The abstraction boundary everyone else depends on.

### `models.py`
```python
class OllamaModel:
    def __init__(self, model: str, host: str) -> None: ...
    def chat(self, messages: list[dict]) -> str: ...
```
- Reads `OLLAMA_MODEL` and `OLLAMA_HOST` from env via constructor args (passed from `main.py`).
- Only file that imports `ollama`.
- `chat()` catches exceptions, logs them, and re-raises.

### `dataset.py`
```python
class DatasetLoader:
    def convert_json_to_jsonl(self, input_file: str, output_file: str) -> None: ...
    def load_jsonl(self, path: str) -> list[dict]: ...

class DatasetValidator:
    def validate(self, dataset: list[dict]) -> dict[str, int]: ...

class TokenCounter:
    def count_dataset_tokens(self, dataset: list[dict]) -> list[int]: ...
    def estimate_epochs(self, n_examples: int) -> int: ...
    def estimate_cost(self, n_epochs: int, token_counts: list[int]) -> float: ...
```
- `DatasetLoader.convert_json_to_jsonl()` raises `FileNotFoundError` if input is missing.
- `DatasetValidator.validate()` returns an error count dict — no side effects, no printing.
- `TokenCounter` encapsulates all epoch/cost estimation constants internally.

### `chatbot.py`
```python
class Conversation:
    def __init__(self, model: ChatModel, system_prompt: str) -> None: ...
    def get_completion(self) -> str: ...
    def collect_messages(self, role: str, message: str) -> None: ...
```
- `model` parameter typed as `ChatModel` protocol — no `ollama` import here.
- Owns the `context` list internally; no global state.
- `get_completion()` returns a fallback string on error so the chat loop never crashes.

### `main.py`
- Instantiates all classes, runs dataset pipeline, then runs the chat loop.
- Contains no business logic — only wiring and I/O.

---

## 4. Data Flow

```
main.py
  │
  ├─▶ DatasetLoader.convert_json_to_jsonl("teacrafter.json", "output.jsonl")
  ├─▶ DatasetLoader.load_jsonl("output.jsonl")           → dataset: list[dict]
  ├─▶ DatasetValidator.validate(dataset)                 → error dict → main prints
  ├─▶ TokenCounter.count_dataset_tokens(dataset)         → token_counts: list[int]
  ├─▶ TokenCounter.estimate_epochs(len(dataset))         → n_epochs: int
  ├─▶ TokenCounter.estimate_cost(n_epochs, token_counts) → cost: float → main prints
  │
  ├─▶ OllamaModel(model=OLLAMA_MODEL, host=OLLAMA_HOST)  → model: ChatModel
  ├─▶ Conversation(model, system_prompt)                 → conv: Conversation
  │
  └─▶ chat loop:
        conv.get_completion()       → calls model.chat(context) → str
        conv.collect_messages()     → appends to internal context
        input("User: ")             → user turn
```

---

## 5. Error Handling

| Location | Behaviour |
|---|---|
| `DatasetLoader.convert_json_to_jsonl()` | Raises `FileNotFoundError` with clear message if input file missing |
| `DatasetValidator.validate()` | Returns error dict; `main.py` prints and decides whether to abort |
| `OllamaModel.chat()` | Catches `Exception`, logs it, re-raises |
| `Conversation.get_completion()` | Catches re-raised exception, returns `"Sorry, I encountered an error."` |

---

## 6. SOLID Mapping

| Principle | Where it lands |
|---|---|
| **S** — Single Responsibility | Each class has one reason to change; each module owns one concern |
| **O** — Open/Closed | Add a new LLM provider by adding a class; no existing code changes |
| **L** — Liskov Substitution | Any `ChatModel` implementor drops into `Conversation` without breakage |
| **I** — Interface Segregation | `ChatModel` protocol has exactly one method — no fat interface |
| **D** — Dependency Inversion | `Conversation` depends on `ChatModel` protocol, not `OllamaModel` directly |

---

## 7. Out of Scope

- No web interface (CLI loop unchanged)
- No new features added beyond what currently exists in `app.py`
- `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `.env` unchanged
