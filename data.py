from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import PreTrainedTokenizer


CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
)


def load_and_sample_dataset(
    dataset_name: str,
    configs: tuple,
    subset_fraction: float = 0.1,
    seed: int = 42,
) -> Dataset:
    dataset_list = []
    for cfg in configs:
        try:
            ds = load_dataset(dataset_name, cfg, split="train")
            dataset_list.append(ds)
        except Exception as e:
            print(f"Could not load config '{cfg}': {e}")

    if not dataset_list:
        raise ValueError("No dataset configs could be loaded.")

    merged = concatenate_datasets(dataset_list)
    num_samples = max(1, int(len(merged) * subset_fraction))
    sampled = merged.shuffle(seed=seed).select(range(num_samples))

    print(f"Loaded {len(merged):,} total examples, using {len(sampled):,} ({subset_fraction*100:.0f}%)")
    return sampled


def _build_prompt_text(messages: list, tokenizer: PreTrainedTokenizer) -> str:
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    if prompt_messages == messages:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )


def _build_full_text(messages: list, tokenizer: PreTrainedTokenizer) -> str:
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, chat_template=CHAT_TEMPLATE
    )


def tokenize_function(examples: dict, tokenizer: PreTrainedTokenizer, max_length: int) -> dict:
    all_input_ids = []
    all_labels = []

    for messages in examples["messages"]:
        full_text = _build_full_text(messages, tokenizer)
        prompt_text = _build_prompt_text(messages, tokenizer)

        full = tokenizer(full_text, truncation=True, max_length=max_length)
        prompt = tokenizer(prompt_text, truncation=True, max_length=max_length)

        input_ids = full["input_ids"]
        prompt_len = min(len(prompt["input_ids"]), len(input_ids))

        labels = input_ids.copy()
        for i in range(prompt_len):
            labels[i] = -100

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    return {"input_ids": all_input_ids, "labels": all_labels}


def prepare_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Dataset:
    fn = lambda examples: tokenize_function(examples, tokenizer, max_length)
    return dataset.map(fn, batched=True, remove_columns=dataset.column_names)
