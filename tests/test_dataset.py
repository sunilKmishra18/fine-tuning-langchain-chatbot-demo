import json
import pytest
from dataset import DatasetLoader


def test_convert_json_to_jsonl_creates_output_file(tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.jsonl"
    data = [
        {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]}
    ]
    input_file.write_text(json.dumps(data))

    DatasetLoader().convert_json_to_jsonl(str(input_file), str(output_file))

    assert output_file.exists()
    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0]) == data[0]


def test_convert_json_to_jsonl_raises_for_missing_file(tmp_path):
    loader = DatasetLoader()
    with pytest.raises(FileNotFoundError):
        loader.convert_json_to_jsonl(
            str(tmp_path / "missing.json"),
            str(tmp_path / "out.jsonl"),
        )


def test_load_jsonl_returns_list_of_dicts(tmp_path):
    jsonl_file = tmp_path / "data.jsonl"
    row = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]}
    jsonl_file.write_text(json.dumps(row) + "\n")

    result = DatasetLoader().load_jsonl(str(jsonl_file))

    assert result == [row]


def test_load_jsonl_raises_for_missing_file(tmp_path):
    loader = DatasetLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_jsonl(str(tmp_path / "missing.jsonl"))


from dataset import DatasetValidator

_VALID_EXAMPLE = {
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
}


def test_validate_valid_dataset_returns_empty_dict():
    errors = DatasetValidator().validate([_VALID_EXAMPLE])
    assert errors == {}


def test_validate_non_dict_entry_counts_data_type_error():
    errors = DatasetValidator().validate(["not a dict"])
    assert errors.get("data_type", 0) == 1


def test_validate_missing_messages_list():
    errors = DatasetValidator().validate([{"no_messages": []}])
    assert errors.get("missing_messages_list", 0) == 1


def test_validate_missing_assistant_message():
    example = {"messages": [{"role": "user", "content": "Hello"}]}
    errors = DatasetValidator().validate([example])
    assert errors.get("example_missing_assistant_message", 0) == 1


def test_validate_message_missing_role_key():
    example = {"messages": [
        {"content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]}
    errors = DatasetValidator().validate([example])
    assert errors.get("message_missing_key", 0) >= 1


from dataset import TokenCounter


def test_count_dataset_tokens_returns_list_of_positive_ints():
    counts = TokenCounter().count_dataset_tokens([_VALID_EXAMPLE])
    assert len(counts) == 1
    assert isinstance(counts[0], int)
    assert counts[0] > 0


def test_estimate_epochs_returns_target_for_medium_dataset():
    # 200 examples * 5 epochs = 1000, between MIN_TARGET (100) and MAX_TARGET (25000)
    assert TokenCounter().estimate_epochs(200) == 5


def test_estimate_epochs_increases_for_tiny_dataset():
    # 1 example * 5 = 5 < 100 MIN_TARGET — epochs must increase
    assert TokenCounter().estimate_epochs(1) > 5


def test_estimate_epochs_decreases_for_huge_dataset():
    # 10000 examples * 5 = 50000 > 25000 MAX_TARGET — epochs must decrease
    assert TokenCounter().estimate_epochs(10_000) < 5


def test_estimate_cost_returns_positive_float():
    cost = TokenCounter().estimate_cost(5, [100, 200, 300])
    assert isinstance(cost, float)
    assert cost > 0
