import re
from datasets import load_dataset
from benchmarks.base import Task


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion: str):
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K(Task):
    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"]
        assert split in ["train", "test"]
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        answer = row["answer"]
        return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}

    def evaluate(self, conversation, assistant_response):
        ref = extract_answer(conversation["messages"][-1]["content"])
        pred = extract_answer(assistant_response)
        return int(pred is not None and pred == ref)
