import os
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    set_seed,
)
import torch
import time


# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000


# TODO: Configure training parameters
TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/gpt2-1b-russian',
    'optim': 'adamw_torch_fused',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 4,
    'save_steps': 100,
    'save_total_limit': 2,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'warmup_steps': 200,
    'logging_steps': 1,
    'eval_steps': 100,
    'eval_strategy': 'steps',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'bf16': True,
    'tf32': True,
    'gradient_checkpointing': False,
    'gradient_accumulation_steps': 1,
    'dataloader_num_workers': 4,
    'torch_compile': True,
    'report_to': 'none',
}


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.stopped_after_seconds = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                if self.stopped_after_seconds is None:
                    self.stopped_after_seconds = elapsed
                control.should_training_stop = True
                # Include the final weights in best-checkpoint selection.
                control.should_evaluate = True
                control.should_save = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    tokenized[LABELS] = [
        [
            token_id if mask == 1 else -100
            for token_id, mask in zip(ids, attention_mask)
        ]
        for ids, attention_mask in zip(
            tokenized[INPUT_IDS],
            tokenized[ATTENTION_MASK],
        )
    ]

    return tokenized


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    os.makedirs(output_dir, exist_ok=True)

    for index in range(num_shards):
        shard = ds.shard(
            num_shards=num_shards,
            index=index,
            contiguous=True,
        )

        path = os.path.join(output_dir, f"{index:05d}.parquet")
        shard.to_parquet(path)


def prepare_dataset(num_proc=8):
    tokenizer = prepare_tokenizer()

    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        writer_batch_size=256,
    )

    save_as_parquets(tokenized_dataset)
    metadata = {
        "dataset": "wikimedia/wikipedia", "configuration": "20231101.ru",
        "rows": len(dataset), "validation_size": VALIDATION_SIZE,
        "tokenizer": TOKENIZER_NAME, "max_length": MAX_LENGTH,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "dataset_fingerprint": dataset._fingerprint,
        "tokenized_fingerprint": tokenized_dataset._fingerprint,
    }
    Path(OUTPUT_DIR, "metadata.json").write_text(json.dumps(metadata, indent=2))



def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    files = sorted(
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"В папке {data_dir} нет Parquet-файлов"
        )

    dataset = load_dataset(
        "parquet",
        data_files={"train": files},
    )

    return dataset["train"]


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def create_model(tokenizer, dtype=torch.bfloat16):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )
    
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='flash_attention_2' if dtype == torch.bfloat16 else 'sdpa',
        torch_dtype=dtype
    )
    
    print(f"Model pad token id: {model.config.pad_token_id}")
    
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
    
    return model


class HistoryCallback(TrainerCallback):
    def __init__(self, folder, timer):
        self.folder, self.timer = Path(folder), timer

    def on_log(self, args, state, control, logs=None, **kwargs):
        row = dict(logs or {})
        if "loss" in row and not math.isfinite(row["loss"]):
            raise FloatingPointError(f"Non-finite training loss: {row}")
        row["step"] = state.global_step
        row["elapsed_seconds"] = (
            time.time() - self.timer.start_time if self.timer.start_time else 0
        )
        with (self.folder / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")


class TimedTrainer(Trainer):
    """Decay LR over the fixed 15-minute budget, rather than a full epoch."""
    def __init__(self, *args, timer, schedule, **kwargs):
        self.timer, self.schedule = timer, schedule
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            def multiplier(step):
                warmup = min((step + 1) / max(self.args.warmup_steps, 1), 1.0)
                if self.schedule == "constant":
                    return warmup
                elapsed = time.time() - self.timer.start_time if self.timer.start_time else 0
                progress = min(max(elapsed / MAX_TRAINING_TIME_SECONDS, 0), 1)
                decay = 1 - progress if self.schedule == "linear" else 0.5 * (1 + math.cos(math.pi * progress))
                return warmup * decay
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer or self.optimizer, multiplier
            )
        return self.lr_scheduler


PROMPTS = [
    "Ахрамеев Денис", "Высшая школа экономики", "Авито",
]


def generate_examples(model, tokenizer, device):
    model.eval()
    examples = []
    with torch.inference_mode():
        for index, prompt in enumerate(PROMPTS):
            set_seed(42 + index)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            for sampling in (False, True):
                options = {"do_sample": sampling}
                if sampling:
                    options.update(temperature=0.8, top_p=0.9)
                generated = model.generate(
                    **inputs, max_new_tokens=80, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, **options,
                )
                examples.append({
                    "prompt": prompt, "sampling": sampling,
                    "text": tokenizer.decode(generated[0], skip_special_tokens=True),
                })
    return examples


def train_model(run_name="baseline", batch_size=8, accumulation=4,
                learning_rate=5e-5, schedule="constant", optim="adamw_torch",
                compile_model=False, bf16=True, tf32=True, smoke_steps=0):
    set_seed(42)
    folder = Path("runs") / run_name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "metrics.json").exists():
        raise FileExistsError(f"Completed run exists: {folder}")
    tokenizer = prepare_tokenizer()
    dataset = load_tokenized_dataset()
    train_dataset, eval_dataset = split_dataset(dataset)
    if smoke_steps:
        train_dataset = train_dataset.select(range(128))
        eval_dataset = eval_dataset.select(range(32))
    model = create_model(tokenizer, dtype=torch.bfloat16 if bf16 else torch.float32)
    # KV cache is for generation; disable it for right-padded training/evaluation.
    model.config.use_cache = False
    parameter_count = sum(p.numel() for p in model.parameters())
    config = dict(TRAINING_CONFIG)
    config.update(
        output_dir=str(folder), per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8, gradient_accumulation_steps=accumulation,
        learning_rate=learning_rate, optim=optim, torch_compile=compile_model,
        bf16=bf16, tf32=tf32, warmup_steps=30, max_steps=smoke_steps or 1000000,
        logging_steps=10 if not smoke_steps else 1, eval_steps=500,
        save_strategy="no", load_best_model_at_end=False, save_only_model=True,
        dataloader_num_workers=4, seed=42, data_seed=42, disable_tqdm=True,
        include_num_input_tokens_seen=True, lr_scheduler_type="constant",
        logging_nan_inf_filter=False,
    )
    timer = TimeoutCallback(MAX_TRAINING_TIME_SECONDS)
    arguments = TrainingArguments(**config)
    trainer = TimedTrainer(
        model=model, args=arguments, train_dataset=train_dataset,
        eval_dataset=eval_dataset, data_collator=default_data_collator,
        callbacks=[timer, HistoryCallback(folder, timer)],
        timer=timer, schedule=schedule,
    )
    (folder / "config.json").write_text(json.dumps({
        **config, "wall_time_schedule": schedule,
        "model_config": model.config.to_dict(), "parameter_count": parameter_count,
        "train_rows": len(train_dataset), "eval_rows": len(eval_dataset),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }, indent=2))
    initial = trainer.evaluate(metric_key_prefix="initial")
    torch.cuda.reset_peak_memory_stats()
    training = trainer.train()
    training_seconds = time.time() - timer.start_time
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    end_evaluations = [row for row in trainer.state.log_history
                       if "eval_loss" in row and row.get("step") == trainer.state.global_step]
    final = ({key: value for key, value in end_evaluations[-1].items() if key.startswith("eval_")}
             if end_evaluations else trainer.evaluate())
    if not math.isfinite(final["eval_loss"]):
        raise FloatingPointError(f"Non-finite final evaluation: {final}")
    trainer.save_state()
    if not smoke_steps:
        trainer.save_model(str(folder / "model"))
        tokenizer.save_pretrained(str(folder / "model"))
        generations = generate_examples(trainer.model, tokenizer, arguments.device)
        (folder / "generations.json").write_text(json.dumps(generations, ensure_ascii=False, indent=2))
    metrics = {
        **initial, **final, **training.metrics,
        "run": run_name, "steps": trainer.state.global_step,
        "training_seconds": training_seconds, "peak_memory_gb": peak_memory_gb,
        "budget_stop_seconds": timer.stopped_after_seconds,
        "input_tokens_seen": trainer.state.num_input_tokens_seen,
        "effective_batch_size": batch_size * accumulation,
        "train_samples_per_second": trainer.state.global_step * batch_size * accumulation / training_seconds,
        "train_steps_per_second": trainer.state.global_step / training_seconds,
        "perplexity": math.exp(final["eval_loss"]), "smoke": bool(smoke_steps),
    }
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("FINAL_METRICS", json.dumps(metrics), flush=True)


def plot_results():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folders = [p.parent for p in sorted(Path("runs").glob("*/metrics.json"))
               if not json.loads(p.read_text()).get("smoke")]
    if not folders:
        raise FileNotFoundError("No completed experiments")
    destination = Path("results")
    destination.mkdir(exist_ok=True)
    colors = ["#2563eb", "#2563eb", "#d97706", "#d97706", "#059669", "#059669", "#be185d", "#be185d"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    all_metrics = []
    for index, folder in enumerate(folders):
        color = colors[index % len(colors)]
        metrics = json.loads((folder / "metrics.json").read_text())
        all_metrics.append(metrics)
        rows = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        training = [row for row in rows if "loss" in row]
        times = np.array([row["elapsed_seconds"] for row in training])
        losses = np.array([row["loss"] for row in training])
        ax.plot(times, losses, color=color, alpha=0.18, linewidth=0.8)
        # The same time window keeps smoothing comparable across different speeds.
        smooth = np.array([losses[(times >= t - 60) & (times <= t)].mean() for t in times])
        ax.plot(times, smooth, color=color, label=folder.name, linewidth=1.7,
                linestyle="-" if index % 2 == 0 else "--")
        single, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        axes[0].plot([r["step"] for r in training], losses, color=color, label="Train loss")
        evaluations = [r for r in rows if "eval_loss" in r or "initial_loss" in r]
        axes[0].plot([r["step"] for r in evaluations],
                     [r.get("eval_loss", r.get("initial_loss")) for r in evaluations],
                     "o--", color="#111827", label="Eval loss (5000 articles)")
        axes[0].set(xlabel="Optimizer step", ylabel="Cross-entropy loss", title=folder.name)
        axes[0].legend()
        rate_rows = [r for r in training if "learning_rate" in r]
        axes[1].plot([r["elapsed_seconds"] for r in rate_rows],
                     [r["learning_rate"] for r in rate_rows], color=color)
        axes[1].set(xlabel="Elapsed time since training start (s)", ylabel="Learning rate", title="Learning-rate schedule")
        for axis in axes:
            axis.grid(alpha=0.2)
        single.savefig(destination / f"{folder.name}_loss.png", dpi=160)
        plt.close(single)
    ax.set(xlabel="Elapsed time since training start (s)", ylabel="Train cross-entropy loss",
           title="900-second budget: train loss (60-second rolling mean)")
    ax.grid(alpha=0.2)
    ax.set_xticks([0, 150, 300, 450, 600, 750, 900])
    ax.legend(fontsize=9)
    fig.savefig(destination / "train_loss_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    names = [m["run"] for m in all_metrics]
    values = [m["eval_loss"] for m in all_metrics]
    bars = ax.barh(names, values, color="#2563eb")
    ax.set(xlabel="Final eval loss on the same first 5000 articles", title="Final evaluation after 15 minutes")
    ax.set_xlim(0, max(values) * 1.15)
    ax.bar_label(bars, fmt="%.4f", padding=4)
    fig.savefig(destination / "eval_loss_comparison.png", dpi=160)
    plt.close(fig)
    (destination / "metrics.json").write_text(json.dumps(all_metrics, indent=2))


def run_experiments():
    Path("runs").mkdir(parents=True, exist_ok=True)
    experiments = [
        ("e1_baseline", []),
        ("e2_learning_rate", ["--learning-rate", "0.0003"]),
        ("e3_cosine", ["--learning-rate", "0.0003", "--schedule", "cosine"]),
        ("e4_microbatch", ["--learning-rate", "0.0003", "--schedule", "cosine", "--batch-size", "16", "--accumulation", "2"]),
        ("e5_fused", ["--learning-rate", "0.0003", "--schedule", "cosine", "--batch-size", "16", "--accumulation", "2", "--optim", "adamw_torch_fused"]),
        ("e6_compile", ["--learning-rate", "0.0003", "--schedule", "cosine", "--batch-size", "16", "--accumulation", "2", "--optim", "adamw_torch_fused", "--compile"]),
        ("e7_fp32_tf32", ["--learning-rate", "0.0003", "--schedule", "cosine", "--batch-size", "16", "--accumulation", "2", "--optim", "adamw_torch_fused", "--fp32"]),
        ("e8_fp32_no_tf32", ["--learning-rate", "0.0003", "--schedule", "cosine", "--batch-size", "16", "--accumulation", "2", "--optim", "adamw_torch_fused", "--fp32", "--no-tf32"]),
    ]
    for name, options in experiments:
        metrics_path = Path("runs", name, "metrics.json")
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("training_seconds", 0) >= MAX_TRAINING_TIME_SECONDS:
                print(f"SKIP completed {name}", flush=True)
                continue
            raise RuntimeError(f"Incomplete result needs inspection: {metrics_path}")
        command = [sys.executable, "-u", __file__, "train", "--run", name, *options]
        print("START", " ".join(command), flush=True)
        with Path("runs", f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("COMPLETE", metrics_path.read_text(), flush=True)
    plot_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "plot", "suite"])
    parser.add_argument("--run", default="baseline")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--schedule", choices=["constant", "linear", "cosine"], default="constant")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_dataset()
    elif args.command == "plot":
        plot_results()
    elif args.command == "suite":
        run_experiments()
    else:
        train_model(args.run, args.batch_size, args.accumulation, args.learning_rate,
                    args.schedule, args.optim, args.compile_model, not args.fp32,
                    not args.no_tf32, args.smoke_steps)
