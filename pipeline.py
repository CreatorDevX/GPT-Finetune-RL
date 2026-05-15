import os
import sys
import time
import json
import torch
from dataclasses import dataclass
from typing import Optional

from config import FinetuneConfig
from data import load_and_sample_dataset, prepare_dataset
from model import load_tokenizer, load_model
from train import train as sft_train


@dataclass
class PipelineConfig:
    model_name: str = "EleutherAI/pythia-1b"
    sft_output_dir: str = "./sft-gpt2xl-smoltalk"
    rl_output_dir: str = "./rl-gpt2xl-gsm8k"
    benchmark_output: str = "./benchmark_results.json"

    # SFT
    sft_batch_size: int = 4
    sft_grad_accum: int = 8
    sft_lr: float = 2e-4
    sft_epochs: int = 1
    sft_max_seq_len: int = 1024
    sft_subset_fraction: float = 0.05
    sft_warmup_ratio: float = 0.03

    # RL
    rl_questions_per_batch: int = 4
    rl_num_samples: int = 8
    rl_max_new_tokens: int = 256
    rl_temperature: float = 0.8
    rl_grad_accum: int = 4
    rl_lr: float = 1e-5
    rl_epochs: int = 1
    rl_warmup_ratio: float = 0.05
    rl_eval_steps: int = 50
    rl_eval_examples: int = 100

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # SVD-Int8
    use_svd_quant: bool = False
    svd_rank: int = 128

    # Benchmark
    bench_max_problems: Optional[int] = None
    bench_batch_size: int = 8

    # System
    mixed_precision: str = "fp16"
    seed: int = 42

    # Resume
    resume_dir: Optional[str] = None


def time_fmt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def phase_header(title):
    w = 76
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)
    print()


def run_sft(cfg: PipelineConfig):
    phase_header(f"PHASE 1: SFT — {cfg.model_name} on SmolTalk → {cfg.sft_output_dir}")

    t_start = time.time()

    os.makedirs(cfg.sft_output_dir, exist_ok=True)

    sft_cfg = FinetuneConfig(
        model_name=cfg.model_name,
        use_lora=not cfg.use_svd_quant,
        use_qlora=False,
        use_svd_quant=cfg.use_svd_quant,
        svd_rank=cfg.svd_rank,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        max_seq_length=cfg.sft_max_seq_len,
        batch_size=cfg.sft_batch_size,
        gradient_accumulation_steps=cfg.sft_grad_accum,
        learning_rate=cfg.sft_lr,
        num_epochs=cfg.sft_epochs,
        warmup_ratio=cfg.sft_warmup_ratio,
        subset_fraction=cfg.sft_subset_fraction,
        output_dir=cfg.sft_output_dir,
        seed=cfg.seed,
        mixed_precision=cfg.mixed_precision,
    )

    print(f"Loading tokenizer...")
    tokenizer = load_tokenizer(sft_cfg)

    print(f"Loading dataset...")
    dataset = load_and_sample_dataset(
        sft_cfg.dataset_name,
        sft_cfg.dataset_configs,
        subset_fraction=sft_cfg.subset_fraction,
        seed=sft_cfg.seed,
    )

    print(f"Tokenizing dataset (max_len={sft_cfg.max_seq_length})...")
    t_tok = time.time()
    tokenized = prepare_dataset(dataset, tokenizer, sft_cfg.max_seq_length)
    print(f"  Tokenized {len(tokenized):,} examples in {time_fmt(time.time() - t_tok)}")

    if cfg.use_svd_quant:
        print(f"Loading model {cfg.model_name} with SVD-Int8 (rank={cfg.svd_rank})...")
    else:
        print(f"Loading model {cfg.model_name} with LoRA (r={cfg.lora_r})...")
    model = load_model(sft_cfg, tokenizer)

    t_train_start = time.time()
    sft_train(sft_cfg, model, tokenizer, tokenized)
    sft_train_time = time.time() - t_train_start
    print(f"  SFT training completed in {time_fmt(sft_train_time)}")

    if cfg.use_svd_quant:
        print(f"Saving SVD-Int8 trained model to {cfg.sft_output_dir}...")
        t_merge = time.time()
        unwrapped = model.merge_and_unload()
        if hasattr(model, 'master_weight_manager'):
            model.master_weight_manager.merge_and_save(model, os.path.join(cfg.sft_output_dir, "merged_master.pt"))
        unwrapped.save_pretrained(cfg.sft_output_dir)
        tokenizer.save_pretrained(cfg.sft_output_dir)
        print(f"  Model saved in {time_fmt(time.time() - t_merge)}")
    else:
        print(f"Merging LoRA and saving merged model to {cfg.sft_output_dir}...")
        t_merge = time.time()
        unwrapped = model.merge_and_unload()
        unwrapped.save_pretrained(cfg.sft_output_dir)
        tokenizer.save_pretrained(cfg.sft_output_dir)
        print(f"  Merged model saved in {time_fmt(time.time() - t_merge)}")

    total_time = time.time() - t_start
    print(f"Phase 1 completed in {time_fmt(total_time)}")
    return {"sft_train_time": sft_train_time, "total_time": total_time}


def run_rl(cfg: PipelineConfig):
    phase_header(f"PHASE 2: RL — GSM8K REINFORCE → {cfg.rl_output_dir}")

    t_start = time.time()
    os.makedirs(cfg.rl_output_dir, exist_ok=True)

    merged_sft_path = os.path.abspath(cfg.sft_output_dir)
    assert os.path.isdir(merged_sft_path), f"SFT merged model not found at {merged_sft_path}"

    print(f"Loaded merged model from {merged_sft_path}")
    print(f"Adding fresh LoRA (r={cfg.lora_r}) for RL...")

    from train_rl import RLConfig, load_model_for_rl, train_rl as rl_train

    rl_cfg = RLConfig(
        num_samples=cfg.rl_num_samples,
        max_new_tokens=cfg.rl_max_new_tokens,
        temperature=cfg.rl_temperature,
        questions_per_batch=cfg.rl_questions_per_batch,
        gradient_accumulation_steps=cfg.rl_grad_accum,
        learning_rate=cfg.rl_lr,
        num_epochs=cfg.rl_epochs,
        warmup_ratio=cfg.rl_warmup_ratio,
        eval_steps=cfg.rl_eval_steps,
        eval_examples=cfg.rl_eval_examples,
        base_model_name=merged_sft_path,
        sft_checkpoint=None,
        use_lora=not cfg.use_svd_quant,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        use_svd_quant=cfg.use_svd_quant,
        svd_rank=cfg.svd_rank,
        output_dir=cfg.rl_output_dir,
        mixed_precision=cfg.mixed_precision,
        seed=cfg.seed + 1,
    )

    t_train_start = time.time()
    rl_train(rl_cfg)
    rl_train_time = time.time() - t_train_start
    print(f"  RL training completed in {time_fmt(rl_train_time)}")

    # Merge RL LoRA into the SFT-merged base and save as fully merged model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Merging RL LoRA into SFT-merged base → {cfg.rl_output_dir}...")
    t_merge = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        merged_sft_path,
        torch_dtype=torch.float16 if cfg.mixed_precision == "fp16" else torch.bfloat16,
        trust_remote_code=True,
    )
    rl_adapter = PeftModel.from_pretrained(base, cfg.rl_output_dir)
    merged = rl_adapter.merge_and_unload()
    merged.save_pretrained(cfg.rl_output_dir)
    tokenizer = AutoTokenizer.from_pretrained(merged_sft_path, trust_remote_code=True)
    tokenizer.save_pretrained(cfg.rl_output_dir)
    print(f"  Fully merged RL model saved in {time_fmt(time.time() - t_merge)}")

    total_time = time.time() - t_start
    print(f"Phase 2 completed in {time_fmt(total_time)}")
    return {"rl_train_time": rl_train_time, "total_time": total_time}


def run_benchmark(cfg: PipelineConfig):
    phase_header(f"PHASE 3: BENCHMARK — full eval suite")

    t_start = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmarks.runner import run_full_benchmark

    merged_path = os.path.abspath(cfg.rl_output_dir)
    assert os.path.isdir(merged_path), f"Fully merged RL model not found at {merged_path}"

    print(f"Loading fully merged RL model from {merged_path}...")
    t_load = time.time()
    torch_dtype = torch.float16 if cfg.mixed_precision == "fp16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype=torch_dtype, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.config.use_cache = False
    print(f"  Loaded in {time_fmt(time.time() - t_load)}")

    print(f"Running benchmarks...")
    results = run_full_benchmark(
        model, tokenizer, device,
        batch_size=cfg.bench_batch_size,
        max_new_tokens=cfg.rl_max_new_tokens,
        max_problems=cfg.bench_max_problems,
    )

    with open(cfg.benchmark_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {cfg.benchmark_output}")

    total_time = time.time() - t_start
    print(f"Phase 3 completed in {time_fmt(total_time)}")
    return {"results": results, "total_time": total_time}


def main():
    cfg = PipelineConfig()

    import argparse
    parser = argparse.ArgumentParser(description="Full SFT → RL → Benchmark pipeline")
    parser.add_argument("--model", default=cfg.model_name, help="Base model name")
    parser.add_argument("--sft-output", default=cfg.sft_output_dir)
    parser.add_argument("--rl-output", default=cfg.rl_output_dir)
    parser.add_argument("--lora-r", type=int, default=cfg.lora_r)
    parser.add_argument("--sft-batch-size", type=int, default=cfg.sft_batch_size)
    parser.add_argument("--sft-grad-accum", type=int, default=cfg.sft_grad_accum)
    parser.add_argument("--sft-epochs", type=int, default=cfg.sft_epochs)
    parser.add_argument("--sft-subset-fraction", type=float, default=cfg.sft_subset_fraction)
    parser.add_argument("--rl-num-samples", type=int, default=cfg.rl_num_samples)
    parser.add_argument("--rl-questions-per-batch", type=int, default=cfg.rl_questions_per_batch)
    parser.add_argument("--rl-epochs", type=int, default=cfg.rl_epochs)
    parser.add_argument("--rl-eval-examples", type=int, default=cfg.rl_eval_examples)
    parser.add_argument("--mixed-precision", default=cfg.mixed_precision)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT, load existing merged model")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL, load existing RL model")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark")
    parser.add_argument("--bench-max-problems", type=int, default=None)
    parser.add_argument("--use-svd-quant", action="store_true", help="Use SVD-Int8 factorization instead of LoRA")
    parser.add_argument("--svd-rank", type=int, default=cfg.svd_rank, help="SVD rank (128 or 256)")
    defaults = vars(parser.parse_args([]))
    args = parser.parse_args()
    for key, val in vars(args).items():
        if val != defaults.get(key):
            setattr(cfg, key, val)

    phases = []
    timings = {}

    if not args.skip_sft:
        timings["sft"] = run_sft(cfg)
        phases.append("sft")
    else:
        print("Skipping SFT phase (--skip-sft)")

    if not args.skip_rl:
        timings["rl"] = run_rl(cfg)
        phases.append("rl")
    else:
        print("Skipping RL phase (--skip-rl)")

    if not args.skip_benchmark:
        timings["benchmark"] = run_benchmark(cfg)
        phases.append("benchmark")
    else:
        print("Skipping benchmark phase (--skip-benchmark)")

    # Summary
    print()
    print("=" * 76)
    print("  PIPELINE COMPLETE")
    print("=" * 76)
    for phase in phases:
        info = timings[phase]
        if phase == "benchmark":
            print(f"  Benchmark results: {json.dumps(info.get('results', {}), indent=4)}")
        else:
            print(f"  {phase.upper()}: {time_fmt(info.get('total_time', 0))} (train: {time_fmt(info.get(f'{phase}_train_time', 0))})")
    total_wall = sum(v["total_time"] for v in timings.values())
    print(f"  Total wall time: {time_fmt(total_wall)}")
    print()

    if not args.skip_benchmark:
        results = timings.get("benchmark", {}).get("results", {})
        if results:
            print(f"  Final ChatCORE: {results.get('ChatCORE', 0):.4f}")
            for task, acc in results.items():
                if task != "ChatCORE":
                    print(f"    {task}: {100 * acc:.2f}%")

    print()
    return timings


if __name__ == "__main__":
    main()
