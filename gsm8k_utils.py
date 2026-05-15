from datasets import load_dataset
from benchmarks.gsm8k import extract_answer


def load_gsm8k(split="train"):
    return load_dataset("openai/gsm8k", "main", split=split)


def format_prompt(question: str, tokenizer):
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )


def compute_reward(generated_text: str, ground_truth: str) -> float:
    pred = extract_answer(generated_text)
    return 1.0 if pred is not None and pred == ground_truth else 0.0
