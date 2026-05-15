import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from typing import Optional

os.environ.setdefault("PEFT_DISABLE_TORCHAO", "1")

from config import FinetuneConfig


def load_tokenizer(config: FinetuneConfig) -> AutoTokenizer:
    model_name = config.model_local_path or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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

    return tokenizer


def _build_quant_config(config: FinetuneConfig) -> Optional[BitsAndBytesConfig]:
    if not config.use_qlora:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(config: FinetuneConfig, tokenizer: Optional[AutoTokenizer] = None) -> torch.nn.Module:
    model_name = config.model_local_path or config.model_name
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32}
    torch_dtype = dtype_map.get(config.mixed_precision, torch.float32)
    quant_config = _build_quant_config(config)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        quantization_config=quant_config,
        device_map="auto" if quant_config else None,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if tokenizer is not None:
        model.resize_token_embeddings(len(tokenizer))

    if config.use_lora:
        if config.target_modules is None:
            config.target_modules = _default_lora_targets(config.model_name)

        if config.use_qlora:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def _default_lora_targets(model_name: str) -> str:
    return "all-linear"


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 150, device="cuda") -> str:
    model.eval()
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=False)
