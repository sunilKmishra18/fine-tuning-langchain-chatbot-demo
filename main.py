import os
from dotenv import load_dotenv, find_dotenv

from models import OllamaModel
from dataset import DatasetLoader, DatasetValidator, TokenCounter
from chatbot import Conversation

load_dotenv(find_dotenv())

_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_SYSTEM_PROMPT = (
    "This is a customer support chatbot designed to help with common "
    "inquiries for TeaCrafters"
)


def run_dataset_pipeline() -> None:
    loader = DatasetLoader()
    loader.convert_json_to_jsonl("teacrafter.json", "output.jsonl")
    dataset = loader.load_jsonl("output.jsonl")

    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)

    errors = DatasetValidator().validate(dataset)
    if errors:
        print("Found errors:")
        for k, v in errors.items():
            print(f"  {k}: {v}")
    else:
        print("No errors found")

    counter = TokenCounter()
    token_counts = counter.count_dataset_tokens(dataset)
    n_epochs = counter.estimate_epochs(len(dataset))
    cost = counter.estimate_cost(n_epochs, token_counts)
    billable = sum(min(TokenCounter.MAX_TOKENS_PER_EXAMPLE, t) for t in token_counts)
    print(f"Dataset has ~{billable} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(f"Estimated cost: ${cost:.4f}")


def run_chat_loop(conv: Conversation) -> None:
    while True:
        conv.collect_messages("assistant", conv.get_completion())
        user_prompt = input("User: ")
        if user_prompt == "exit":
            print("\nGoodbye")
            break
        conv.collect_messages("user", user_prompt)


if __name__ == "__main__":
    try:
        run_dataset_pipeline()
        model = OllamaModel(model=_OLLAMA_MODEL, host=_OLLAMA_HOST)
        conv = Conversation(model=model, system_prompt=_SYSTEM_PROMPT)
        run_chat_loop(conv)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise SystemExit(1)
