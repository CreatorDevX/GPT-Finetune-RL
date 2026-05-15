import torch
from benchmarks.gsm8k import GSM8K
from benchmarks.mmlu import MMLU
from benchmarks.arc import ARC
from benchmarks.humaneval import HumanEval

TASK_REGISTRY = {
    "GSM8K": lambda: GSM8K(subset="main", split="test"),
    "MMLU": lambda: MMLU(subset="all", split="test"),
    "ARC-Easy": lambda: ARC(subset="ARC-Easy", split="test"),
    "ARC-Challenge": lambda: ARC(subset="ARC-Challenge", split="test"),
    "HumanEval": lambda: HumanEval(),
}

BASELINE_ACCURACIES = {
    "GSM8K": 0.0,
    "MMLU": 0.25,
    "ARC-Easy": 0.25,
    "ARC-Challenge": 0.25,
    "HumanEval": 0.0,
}


def format_prompt(conversation, tokenizer):
    user_messages = [m for m in conversation["messages"] if m["role"] != "assistant"]
    return tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)


def run_generative_eval(task, model, tokenizer, device, max_new_tokens=256, temperature=0.0, top_k=50, max_problems=None, num_samples=1):
    model.eval()
    num_problems = min(len(task), max_problems) if max_problems else len(task)
    num_passed = 0

    for i in range(num_problems):
        conversation = task[i]
        prompt = format_prompt(conversation, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=False)
        is_correct = task.evaluate(conversation, gen_text)
        num_passed += int(is_correct)

    return num_passed / max(num_problems, 1)


def run_categorical_eval(task, model, tokenizer, device, batch_size=8, max_problems=None):
    model.eval()
    num_problems = min(len(task), max_problems) if max_problems else len(task)
    letter_to_id = {}
    num_passed = 0
    total = 0

    for i in range(0, num_problems, batch_size):
        batch_indices = list(range(i, min(i + batch_size, num_problems)))
        conversations = [task[idx] for idx in batch_indices]
        prompts = [format_prompt(conv, tokenizer) for conv in conversations]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        for j, conversation in enumerate(conversations):
            letters = conversation["letters"]
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id:
                    encoded = tokenizer.encode(letter, add_special_tokens=False)
                    assert len(encoded) == 1
                    letter_to_id[letter] = encoded[0]
                letter_ids.append(letter_to_id[letter])

            answer_pos = (inputs["attention_mask"][j].sum() - 1).item()
            focus_logits = logits[j, answer_pos, letter_ids]
            predicted_letter = letters[focus_logits.argmax().item()]
            outcome = task.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    return num_passed / max(total, 1)


def run_evaluation(task_name, model, tokenizer, device, batch_size=8, max_new_tokens=256, temperature=0.0, top_k=50, max_problems=None, num_samples=1):
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")

    task = TASK_REGISTRY[task_name]()

    if task.eval_type == "generative":
        acc = run_generative_eval(task, model, tokenizer, device, max_new_tokens, temperature, top_k, max_problems, num_samples)
    elif task.eval_type == "categorical":
        acc = run_categorical_eval(task, model, tokenizer, device, batch_size, max_problems)
    else:
        raise ValueError(f"Unknown eval type: {task.eval_type}")

    return acc


def run_full_benchmark(model, tokenizer, device, **kwargs):
    results = {}
    for task_name in TASK_REGISTRY:
        acc = run_evaluation(task_name, model, tokenizer, device, **kwargs)
        results[task_name] = acc
        print(f"  {task_name}: {100 * acc:.2f}%")

    # ChatCORE: mean centered accuracy
    centered_scores = []
    for task_name, acc in results.items():
        baseline = BASELINE_ACCURACIES.get(task_name, 0.0)
        centered = (acc - baseline) / (1.0 - baseline + 1e-10)
        centered_scores.append(centered)
    chatcore = sum(centered_scores) / len(centered_scores)
    results["ChatCORE"] = chatcore
    print(f"  ChatCORE: {chatcore:.4f}")

    return results
