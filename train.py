import os
import math
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler, set_seed
from tqdm.auto import tqdm
from accelerate import Accelerator

from config import FinetuneConfig


def train(config: FinetuneConfig, model, tokenizer, train_dataset):
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )

    set_seed(config.seed)

    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

        padded = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        )
        padded_labels = tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        )
        padded_labels["input_ids"][padded_labels["attention_mask"] == 0] = -100

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": padded_labels["input_ids"],
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    total_steps = len(train_dataloader) * config.num_epochs
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    accumulated_loss = 0.0
    model.train()

    for epoch in range(config.num_epochs):
        per_epoch_steps = (len(train_dataloader) + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps
        progress_bar = tqdm(
            range(per_epoch_steps),
            desc=f"Epoch {epoch+1}/{config.num_epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                accumulated_loss += loss.detach().float()
                global_step += 1
                progress_bar.update(1)

                if global_step % config.logging_steps == 0:
                    avg_loss = accumulated_loss / config.logging_steps
                    ppl = math.exp(min(avg_loss, 20))
                    lr = lr_scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "ppl": f"{ppl:.2f}",
                        "lr": f"{lr:.2e}",
                    })
                    accumulated_loss = 0.0

                if global_step % config.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        ckpt_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        accelerator.unwrap_model(model).save_pretrained(ckpt_path)
                        tokenizer.save_pretrained(ckpt_path)
                        print(f"Checkpoint saved: {ckpt_path}")

        progress_bar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        print(f"Model saved: {config.output_dir}")
