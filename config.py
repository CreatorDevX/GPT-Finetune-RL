from dataclasses import dataclass
from typing import Optional


@dataclass
class FinetuneConfig:
    model_name: str = "gpt2-xl"
    model_local_path: Optional[str] = None
    use_lora: bool = True
    use_qlora: bool = False  # 4-bit quantization (requires bitsandbytes)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[list] = None

    dataset_name: str = "HuggingFaceTB/smoltalk"
    dataset_configs: tuple = (
        "everyday-conversations",
        "model-instruct",
        "multi-turn-conversations",
    )
    subset_fraction: float = 0.1

    max_seq_length: int = 1024
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 1
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    output_dir: str = "./gpt2-xl-smoltalk"
    seed: int = 42
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"  # "fp16", "bf16", or "no"
