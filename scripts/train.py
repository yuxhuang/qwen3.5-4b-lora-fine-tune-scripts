#!/usr/bin/env python3
"""
LoRA SFT training for Qwen agentic loop. Uses config/lora_sft.yaml and data from data/train.jsonl, data/eval.jsonl.
Run from project root with: conda activate qwen-agentic-finetune && python scripts/train.py
"""

import json
import os
import shutil
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "lora_sft.yaml"
DATA_DIR = PROJECT_ROOT / "data"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def conversations_to_messages(example):
    """Convert conversations (from/value) to messages (role/content) for chat template."""
    convs = example["conversations"]
    user_content = None
    assistant_content = None
    for c in convs:
        if c["from"] == "user":
            user_content = c["value"]
        elif c["from"] == "assistant":
            assistant_content = c["value"]
    if user_content is None or assistant_content is None:
        raise ValueError("Each example must have exactly one user and one assistant message.")
    return {
        "messages": [
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": assistant_content.strip()},
        ]
    }


def tokenize_with_chat_template(example, tokenizer, max_length):
    """
    Build full sequence from messages using the tokenizer's chat template, then tokenize once.
    Returns input_ids and completion_mask so prompt/completion tokenization is consistent
    (avoids TRL 'Mismatch between tokenized prompt and ...' warning).
    """
    messages = example["messages"]
    # Full sequence: user + assistant (same as template output for [user, assistant])
    full_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    # Prompt part: user turn only (exact prefix of full_str)
    prompt_str = tokenizer.apply_chat_template(
        [messages[0]],
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer.encode(full_str, add_special_tokens=False)
    prompt_len = len(tokenizer.encode(prompt_str, add_special_tokens=False))
    completion_mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        completion_mask = completion_mask[:max_length]
    return {"input_ids": full_ids, "completion_mask": completion_mask}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap train set size (for smoke test).")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_train_epochs.")
    args_cli = parser.parse_args()

    cfg = load_config()
    model_name = cfg["model"]["name_or_path"]
    train_file = PROJECT_ROOT / cfg["data"]["train_file"]
    eval_file = PROJECT_ROOT / cfg["data"]["eval_file"]
    out_dir = PROJECT_ROOT / cfg["training"]["output_dir"]
    max_length = cfg["training"]["max_seq_length"]
    if args_cli.epochs is not None:
        cfg["training"]["num_train_epochs"] = args_cli.epochs

    # Load tokenizer first so we can pre-tokenize with the chat template (avoids prompt/completion mismatch).
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data: jsonl with "conversations" -> messages -> input_ids + completion_mask
    train_ds = load_dataset("json", data_files=str(train_file), split="train")
    if args_cli.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args_cli.max_train_samples, len(train_ds))))
    eval_ds = load_dataset("json", data_files=str(eval_file), split="train") if eval_file.exists() else None
    if eval_ds is not None and args_cli.max_train_samples is not None:
        eval_ds = eval_ds.select(range(min(4, len(eval_ds))))

    train_ds = train_ds.map(
        conversations_to_messages,
        remove_columns=train_ds.column_names,
        num_proc=4,
        desc="Convert to messages",
    )
    train_ds = train_ds.map(
        tokenize_with_chat_template,
        remove_columns=train_ds.column_names,
        num_proc=4,
        desc="Tokenize with chat template",
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
    )
    if eval_ds is not None:
        eval_ds = eval_ds.map(
            conversations_to_messages,
            remove_columns=eval_ds.column_names,
            num_proc=4,
            desc="Convert to messages",
        )
        eval_ds = eval_ds.map(
            tokenize_with_chat_template,
            remove_columns=eval_ds.column_names,
            num_proc=4,
            desc="Tokenize with chat template",
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        )

    # GPU-only training: require a usable GPU unless ALLOW_CPU_TRAIN=1 (debug only).
    allow_cpu = os.environ.get("ALLOW_CPU_TRAIN", "") == "1"
    use_cuda = False
    if allow_cpu and not torch.cuda.is_available():
        use_cuda = False
    else:
        try:
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                torch.zeros(1).cuda()  # trigger kernel load; may raise if device not supported
        except Exception:
            use_cuda = False
    if not use_cuda:
        if not allow_cpu:
            print("Training requires a GPU. CUDA is not available or GPU failed.", file=sys.stderr)
            print("See docs/MACHINE_AND_SETUP.md for setup. Set ALLOW_CPU_TRAIN=1 for CPU debug only.", file=sys.stderr)
            sys.exit(1)
        print("ALLOW_CPU_TRAIN=1: running on CPU (debug only, not for real training).")
    device_map = "auto" if use_cuda else "cpu"
    use_cpu = device_map == "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device_map,
    )

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)

    training = cfg["training"]
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=training["num_train_epochs"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training.get("per_device_eval_batch_size", training["per_device_train_batch_size"]),
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        warmup_ratio=training["warmup_ratio"],
        lr_scheduler_type=training["lr_scheduler_type"],
        logging_steps=training["logging_steps"],
        eval_strategy=training["eval_strategy"],
        save_strategy=training["save_strategy"],
        save_total_limit=training["save_total_limit"],
        bf16=training["bf16"] and not use_cpu,
        fp16=False,
        use_cpu=use_cpu,  # True only when ALLOW_CPU_TRAIN=1 (debug)
        gradient_checkpointing=training.get("gradient_checkpointing", False),
        max_length=training["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    final_dir = out_dir / "final"
    save_final(trainer, tokenizer, final_dir)


def save_final(trainer, tokenizer, final_dir: Path):
    """Save final adapter, tokenizer, trainer state, and a metadata file to final_dir.

    Copies trainer_state.json from the last checkpoint so the final output
    is self-contained and verifiable.  Writes a metadata.json with the
    timestamp and source checkpoint step for auditability.
    """
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    state = trainer.state
    last_checkpoint = None
    if state.best_model_checkpoint:
        last_checkpoint = Path(state.best_model_checkpoint)
    else:
        out_dir = Path(trainer.args.output_dir)
        candidates = sorted(out_dir.glob("checkpoint-*"), key=os.path.getmtime)
        if candidates:
            last_checkpoint = candidates[-1]

    if last_checkpoint and (last_checkpoint / "trainer_state.json").exists():
        shutil.copy2(last_checkpoint / "trainer_state.json", final_dir / "trainer_state.json")

    metadata = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "global_step": state.global_step,
        "epoch": state.epoch,
        "source_checkpoint": str(last_checkpoint) if last_checkpoint else None,
        "train_loss": getattr(state, "training_loss", None),
    }
    eval_entries = [e for e in (state.log_history or []) if "eval_loss" in e]
    if eval_entries:
        last_eval = eval_entries[-1]
        metadata["eval_loss"] = last_eval.get("eval_loss")
        metadata["eval_mean_token_accuracy"] = last_eval.get("eval_mean_token_accuracy")

    with open(final_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"Final adapter saved to {final_dir}")
    print(f"  step={metadata['global_step']}, epoch={metadata['epoch']}")
    if metadata.get("eval_loss"):
        print(f"  eval_loss={metadata['eval_loss']:.4f}, eval_acc={metadata['eval_mean_token_accuracy']:.4f}")
    if last_checkpoint:
        print(f"  trainer_state.json copied from {last_checkpoint.name}")


if __name__ == "__main__":
    main()
