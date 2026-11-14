"""One RL experiment with TRL GRPOTrainer; never performs supervised fine-tuning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from prepare_data import MODEL, MODEL_REVISION, SEED
from reward import concise_no_call_reward, dense_reward, strict_reward


def last_complete_checkpoint(directory: Path):
    paths = sorted(
        directory.glob("checkpoint-*"),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
        reverse=True,
    )
    for path in paths:
        if all((path / name).is_file() for name in (
            "trainer_state.json", "adapter_model.safetensors", "optimizer.pt"
        )):
            return str(path)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train", default="data/train.jsonl", type=Path)
    parser.add_argument("--validation", default="data/validation.jsonl", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    started_file = args.output / "started_at.json"
    if not started_file.exists():
        started_file.write_text(json.dumps({"unix_seconds": time.time()}) + "\n")
    started_at = json.loads(started_file.read_text())["unix_seconds"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_path = Path(config.get("train_file", str(args.train)))
    train = Dataset.from_json(str(train_path))
    validation = Dataset.from_json(str(args.validation))
    lora = LoraConfig(
        r=config["lora_r"],
        lora_alpha=2 * config["lora_r"],
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    training = GRPOConfig(
        output_dir=str(args.output / "trainer"),
        model_init_kwargs={"revision": MODEL_REVISION, "dtype": "bfloat16"},
        seed=SEED,
        data_seed=SEED,
        bf16=True,
        max_steps=config["max_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_generations=4,
        generation_batch_size=16,
        max_completion_length=256,
        temperature=0.8,
        beta=0.0,
        loss_type=config["loss_type"],
        scale_rewards=config["scale_rewards"],
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=2048,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        logging_steps=1,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        log_completions=False,
    )
    reward_fn = {
        "dense": dense_reward,
        "strict": strict_reward,
        "concise_no_call": concise_no_call_reward,
    }[config["reward"]]
    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=reward_fn,
        args=training,
        train_dataset=train,
        eval_dataset=validation,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    checkpoint = last_complete_checkpoint(args.output / "trainer")
    if checkpoint:
        print(f"Resuming from {checkpoint}", flush=True)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    active_seconds = round(time.monotonic() - start, 1)
    seconds = round(time.time() - started_at, 1)
    peak_gib = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    history = trainer.state.log_history
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output / "training.json").write_text(json.dumps({
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "seconds": seconds,
        "active_seconds_last_attempt": active_seconds,
        "peak_allocated_gib": peak_gib,
        "train_loss": result.training_loss,
        "step": trainer.state.global_step,
        "train_samples": len(train),
        "train_file": str(train_path),
        "validation_samples": len(validation),
    }, indent=2) + "\n")
    adapter = args.output / "adapter"
    trainer.model.save_pretrained(adapter)
    merged = trainer.model.merge_and_unload()
    merged_dir = args.output / "merged"
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(merged_dir)
    print(f"Saved merged checkpoint: {merged_dir}")


if __name__ == "__main__":
    main()
