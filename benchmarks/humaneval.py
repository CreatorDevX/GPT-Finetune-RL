import re
from datasets import load_dataset
from benchmarks.base import Task
from benchmarks.execution import execute_code


def extract_imports(prompt):
    imports = []
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
        elif stripped and not stripped.startswith("#"):
            break
    return "\n".join(imports)


def extract_program(completion):
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        return matches[0].strip()
    return completion.strip()


class HumanEval(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt = row["prompt"]
        solution = row["canonical_solution"]
        entry_point = row["entry_point"]
        test = row["test"]
        complete_solution = f"{prompt}\n{solution}"
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": complete_solution},
            ],
            "entry_point": entry_point,
            "test": test,
        }

    def evaluate(self, conversation, completion):
        prompt = conversation["messages"][0]["content"]
        imports = extract_imports(prompt)
        completion_code = extract_program(completion)
        program = (
            imports
            + "\n\n"
            + completion_code
            + "\n\n"
            + conversation["test"]
            + "\n"
            + f"check({conversation['entry_point']})"
        )
        result = execute_code(program)
        return result.success
