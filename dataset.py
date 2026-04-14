import json
import os
import tiktoken
from collections import defaultdict

_encoding = tiktoken.get_encoding("cl100k_base")


class DatasetLoader:
    def convert_json_to_jsonl(self, input_file: str, output_file: str) -> None:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        with open(input_file) as f:
            data = json.load(f)
        with open(output_file, "w") as outfile:
            for entry in data:
                json.dump(entry, outfile)
                outfile.write("\n")

    def load_jsonl(self, path: str) -> list[dict]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSONL file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


class DatasetValidator:
    _VALID_ROLES = {"system", "user", "assistant", "function"}
    _VALID_KEYS = {"role", "content", "name", "function_call"}

    def validate(self, dataset: list[dict]) -> dict[str, int]:
        errors: dict[str, int] = defaultdict(int)
        for ex in dataset:
            if not isinstance(ex, dict):
                errors["data_type"] += 1
                continue
            messages = ex.get("messages")
            if not messages:
                errors["missing_messages_list"] += 1
                continue
            for msg in messages:
                if "role" not in msg or "content" not in msg:
                    errors["message_missing_key"] += 1
                if any(k not in self._VALID_KEYS for k in msg):
                    errors["message_unrecognized_key"] += 1
                if msg.get("role") not in self._VALID_ROLES:
                    errors["unrecognized_role"] += 1
                content = msg.get("content")
                function_call = msg.get("function_call")
                if (not content and not function_call) or not isinstance(content, str):
                    errors["missing_content"] += 1
            if not any(msg.get("role") == "assistant" for msg in messages):
                errors["example_missing_assistant_message"] += 1
        return dict(errors)


class TokenCounter:
    MAX_TOKENS_PER_EXAMPLE = 4096
    TARGET_EPOCHS = 5
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25
    COST_PER_1K_TOKENS = 0.0080

    def _tokens_in_messages(self, messages: list[dict]) -> int:
        num_tokens = 3  # every reply is primed with <im_start>assistant
        for msg in messages:
            num_tokens += 3  # role/name/content framing
            for key, value in msg.items():
                num_tokens += len(_encoding.encode(value))
                if key == "name":
                    num_tokens += 1
        return num_tokens

    def count_dataset_tokens(self, dataset: list[dict]) -> list[int]:
        return [self._tokens_in_messages(ex["messages"]) for ex in dataset]

    def estimate_epochs(self, n_examples: int) -> int:
        n_epochs = self.TARGET_EPOCHS
        if n_examples * self.TARGET_EPOCHS < self.MIN_TARGET_EXAMPLES:
            n_epochs = min(
                self.MAX_DEFAULT_EPOCHS,
                self.MIN_TARGET_EXAMPLES // n_examples,
            )
        elif n_examples * self.TARGET_EPOCHS > self.MAX_TARGET_EXAMPLES:
            n_epochs = max(
                self.MIN_DEFAULT_EPOCHS,
                self.MAX_TARGET_EXAMPLES // n_examples,
            )
        return n_epochs

    def estimate_cost(self, n_epochs: int, token_counts: list[int]) -> float:
        billable = sum(
            min(self.MAX_TOKENS_PER_EXAMPLE, t) for t in token_counts
        )
        return (n_epochs * billable / 1000) * self.COST_PER_1K_TOKENS
