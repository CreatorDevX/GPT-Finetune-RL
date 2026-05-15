import os
import math
import random
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import get_scheduler, set_seed, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Optional

from gsm8k_utils import extract_answer, load_gsm8k, format_prompt, compute_reward


def init_wandb(config: RLConfig, accelerator: Accelerator):
    try:
        import wandb
        if accelerator.is_main_process:
            wandb.init(
                project="rl-reinforce-gsm8k",
                config={
                    "base_model_name": config.base_model_name,
                    "sft_checkpoint": config.sft_checkpoint,
                    "use_lora": config.use_lora,
                    "use_svd_quant": config.use_svd_quant,
                    "svd_rank": config.svd_rank,
                    "lora_r": config.lora_r,
                    "lora_alpha": config.lora_alpha,
                    "lora_dropout": config.lora_dropout,
                    "num_samples": config.num_samples,
                    "max_new_tokens": config.max_new_tokens,
                    "temperature": config.temperature,
                    "top_k": config.top_k,
                    "questions_per_batch": config.questions_per_batch,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "learning_rate": config.learning_rate,
                    "num_epochs": config.num_epochs,
                    "warmup_ratio": config.warmup_ratio,
                    "mixed_precision": config.mixed_precision,
                    "seed": config.seed,
                    "max_grad_norm": config.max_grad_norm,
                    "eval_steps": config.eval_steps,
                    "eval_examples": config.eval_examples,
                },
            )
        return True
    except ImportError:
        return False


def log_wandb(metrics: dict, step: int):
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


@dataclass
class RLConfig:
    # Rollout generation
    num_samples: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50

    # Training
    questions_per_batch: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05

    # Logging / saving
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 200
    output_dir: str = "./gsm8k-rl"
    eval_examples: int = 100

    # Model
    base_model_name: str = "EleutherAI/pythia-1b"
    sft_checkpoint: Optional[str] = "./gpt2-xl-smoltalk"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[list | str] = None

    use_svd_quant: bool = False
    svd_rank: int = 128

    # System
    mixed_precision: str = "bf16"
    seed: int = 42
    max_grad_norm: float = 1.0


def _default_lora_targets(model_name: str) -> str:
    return "all-linear"


def load_model_for_rl(config: RLConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, revision="step10000", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        revision="step10000",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if config.use_svd_quant:
        from svd_quant import apply_svd_to_model, MasterWeightManager, print_svd_info
        print("Applying SVD factorization to base weights...")
        model = apply_svd_to_model(model, rank=config.svd_rank)
        print_svd_info(model)
        master_mgr = MasterWeightManager(model, os.path.join(config.output_dir, "master_weights.pt"))
        model.master_weight_manager = master_mgr
    else:
        loaded_sft = False
        if config.sft_checkpoint is not None and os.path.isdir(config.sft_checkpoint):
            model = PeftModel.from_pretrained(model, config.sft_checkpoint)
            print(f"Loaded SFT adapter from {config.sft_checkpoint}")
            loaded_sft = True

        if config.use_lora and not loaded_sft:
            targets = config.target_modules
            if targets is None:
                targets = _default_lora_targets(config.base_model_name)
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=targets,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    return model, tokenizer


def generate_rollouts(model, tokenizer, questions, ground_truths, config: RLConfig, device):
    model.eval()
    all_rollouts = []
    all_rewards = []

    for q_idx, (question, gt) in enumerate(zip(questions, ground_truths)):
        prompt_ids = format_prompt(question, tokenizer)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_len = len(prompt_ids)

        for s_idx in range(config.num_samples):
            with torch.no_grad():
                full_ids = model.generate(
                    prompt_tensor,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            gen_ids = full_ids[0, prompt_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            reward = compute_reward(gen_text, gt)

            full_seq = full_ids[0]
            labels = full_seq.clone()
            labels[:prompt_len] = -100

            all_rollouts.append({
                "input_ids": full_seq,
                "labels": labels,
                "reward": reward,
                "seq_len": len(full_seq),
            })
            all_rewards.append(reward)

    rewards_t = torch.tensor(all_rewards, dtype=torch.float, device=device)
    mean_r = rewards_t.mean()
    std_r = rewards_t.std()
    if std_r < 1e-8:
        advantages = rewards_t - mean_r
    else:
        advantages = (rewards_t - mean_r) / (std_r + 1e-8)

    return all_rollouts, advantages, all_rewards


def collate_rollouts(rollouts, advantages, pad_token_id, device):
    max_len = max(r["seq_len"] for r in rollouts)
    B = len(rollouts)

    padded_inputs = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
    padded_labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)

    for i, r in enumerate(rollouts):
        slen = r["seq_len"]
        padded_inputs[i, :slen] = r["input_ids"]
        padded_labels[i, :slen] = r["labels"]

    inputs = padded_inputs[:, :-1]
    targets = padded_labels[:, 1:]
    advs = advantages

    return inputs, targets, advs


def compute_pg_loss(model, inputs, targets, advantages):
    outputs = model(input_ids=inputs)
    logits = outputs.logits

    mask = (targets >= 0).float()
    targets_clamped = targets.clamp(min=0)

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets_clamped.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * mask

    adv = advantages.unsqueeze(-1)
    pg_obj = (token_log_probs * adv).sum()
    num_valid = mask.sum().clamp(min=1)
    loss = -pg_obj / num_valid
    return loss


@torch.no_grad()
def evaluate_gsm8k(model, tokenizer, config: RLConfig, device, num_examples=100):
    model.eval()
    ds = load_gsm8k(split="test")
    correct = 0
    total = 0
    for idx in range(min(num_examples, len(ds))):
        row = ds[idx]
        question = row["question"]
        gt = extract_answer(row["answer"])
        if gt is None:
            continue

        prompt_ids = format_prompt(question, tokenizer)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_len = len(prompt_ids)

        full_ids = model.generate(
            prompt_tensor,
            max_new_tokens=config.max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        gen_text = tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=False)
        pred = extract_answer(gen_text)
        if pred == gt:
            correct += 1
        total += 1

    acc = correct / max(total, 1)
    return acc


def train_rl(config: RLConfig):
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    device = accelerator.device
    set_seed(config.seed)

    model, tokenizer = load_model_for_rl(config)

    train_ds = load_gsm8k(split="train")
    print(f"Loaded GSM8K train: {len(train_ds)} examples")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    steps_per_epoch = len(train_ds) // config.questions_per_batch
    total_steps = steps_per_epoch * config.num_epochs
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    use_wandb = init_wandb(config, accelerator)

    global_step = 0
    accumulated_loss = 0.0
    best_eval_acc = 0.0
    progress_bar = tqdm(range(total_steps), desc="RL Training", disable=not accelerator.is_local_main_process)

    for epoch in range(config.num_epochs):
        indices = list(range(len(train_ds)))
        set_seed(config.seed + epoch)
        random.shuffle(indices)

        for batch_start in range(0, len(train_ds), config.questions_per_batch):
            if global_step >= total_steps:
                break

            batch_indices = indices[batch_start:batch_start + config.questions_per_batch]
            questions = [train_ds[i]["question"] for i in batch_indices]
            ground_truths = [extract_answer(train_ds[i]["answer"]) for i in batch_indices]

            rollouts, advantages, rewards = generate_rollouts(
                accelerator.unwrap_model(model), tokenizer,
                questions, ground_truths, config, device
            )
            inputs, targets, advs = collate_rollouts(
                rollouts, advantages, tokenizer.pad_token_id, device
            )

            with accelerator.accumulate(model):
                loss = compute_pg_loss(model, inputs, targets, advs)
                accelerator.backward(loss)

                grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                accumulated_loss += loss.detach().float()
                global_step += 1
                progress_bar.update(1)

                if global_step % config.logging_steps == 0:
                    avg_loss = accumulated_loss / config.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                    })
                    tqdm.write(f"Step {global_step:05d} | loss: {avg_loss:.4f}")

                    if use_wandb:
                        reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
                        reward_std = (sum((r - reward_mean) ** 2 for r in rewards) / max(len(rewards), 1)) ** 0.5
                        reward_max = max(rewards) if rewards else 0.0
                        reward_min = min(rewards) if rewards else 0.0
                        log_wandb({
                            "rl/loss": avg_loss,
                            "rl/learning_rate": lr,
                            "rl/gradient_norm": grad_norm,
                            "rl/epoch": epoch,
                            "rl/global_step": global_step,
                            "rl/progress": global_step / max(total_steps, 1),
                            "rl/reward_mean": reward_mean,
                            "rl/reward_std": reward_std,
                            "rl/reward_max": reward_max,
                            "rl/reward_min": reward_min,
                            "rl/advantage_mean": advs.mean().item(),
                            "rl/advantage_std": advs.std().item(),
                        }, step=global_step)

                    accumulated_loss = 0.0

            if accelerator.sync_gradients and global_step % config.eval_steps == 0:
                unwrapped = accelerator.unwrap_model(model)
                eval_acc = evaluate_gsm8k(
                    unwrapped, tokenizer, config, device,
                    num_examples=config.eval_examples
                )
                tqdm.write(f"Step {global_step:05d} | Eval accuracy (GSM8K): {eval_acc:.4f}")

                if use_wandb:
                    log_wandb({
                        "rl/eval_accuracy": eval_acc,
                        "rl/best_eval_accuracy": max(eval_acc, best_eval_acc),
                        "rl/eval_step": global_step,
                    }, step=global_step)

                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    os.makedirs(config.output_dir, exist_ok=True)
                    unwrapped.save_pretrained(config.output_dir)
                    tokenizer.save_pretrained(config.output_dir)
                    tqdm.write(f"  -> Saved best model to {config.output_dir}")

                    if use_wandb:
                        log_wandb({"rl/best_model_saved": True}, step=global_step)

            if accelerator.sync_gradients and global_step % config.save_steps == 0:
                unwrapped = accelerator.unwrap_model(model)
                ckpt_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_path, exist_ok=True)
                unwrapped.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                tqdm.write(f"Checkpoint saved: {ckpt_path}")

                if use_wandb:
                    log_wandb({"rl/checkpoint_saved": ckpt_path}, step=global_step)

    progress_bar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        if config.use_svd_quant and hasattr(unwrapped, 'master_weight_manager'):
            merged_path = os.path.join(config.output_dir, "merged_master.pt")
            unwrapped.master_weight_manager.merge_and_save(unwrapped, merged_path)
            print(f"Master weights merged and saved: {merged_path}")
        unwrapped.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        print(f"Final model saved: {config.output_dir}")
        print(f"Best eval accuracy: {best_eval_acc:.4f}")

        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    config = RLConfig()
    train_rl(config)
