#!/usr/bin/env python3
"""
Run evaluation: load trained (or base) model and compute eval loss on data/eval.jsonl.
Optionally run a short agent loop on one example with tools_executor (requires beautifulsoup4).
Usage:
  conda activate qwen-agentic-finetune
  python scripts/run_agent_eval.py --checkpoint outputs/lora_sft/final
  python scripts/run_agent_eval.py --checkpoint outputs/lora_sft/final --run-agent-one
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def conversations_to_prompt_completion(example):
    convs = example["conversations"]
    user_content = assistant_content = None
    for c in convs:
        if c["from"] == "user":
            user_content = c["value"]
        elif c["from"] == "assistant":
            assistant_content = c["value"]
    return {
        "prompt": [{"role": "user", "content": user_content}],
        "completion": [{"role": "assistant", "content": assistant_content}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to adapter or full model (default: base model).")
    parser.add_argument("--eval-file", type=Path, default=PROJECT_ROOT / "data" / "eval.jsonl", help="Eval jsonl file.")
    parser.add_argument("--run-agent-one", action="store_true", help="Run one agent loop with tools (sanity check).")
    args = parser.parse_args()

    # Resolve model: checkpoint or default base
    if args.checkpoint:
        model_path = Path(args.checkpoint)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
    else:
        model_path = "Qwen/Qwen3.5-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_path if isinstance(model_path, str) else str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path if isinstance(model_path, str) else str(model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    if not args.eval_file.exists():
        print(f"Eval file not found: {args.eval_file}. Run scripts/generate_data.py first.")
        return

    eval_ds = load_dataset("json", data_files=str(args.eval_file), split="train")
    eval_ds = eval_ds.map(conversations_to_prompt_completion, remove_columns=eval_ds.column_names)

    total_loss = 0.0
    n = 0
    max_length = 2048

    for example in eval_ds:
        prompt = example["prompt"]
        completion = example["completion"]
        full_messages = prompt + completion
        text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"].clone())
        total_loss += out.loss.item()
        n += 1

    if n > 0:
        avg_loss = total_loss / n
        try:
            ppl = torch.exp(torch.tensor(avg_loss)).item()
        except Exception:
            ppl = float("inf")
        print(f"Eval examples: {n}")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Perplexity: {ppl:.4f}")

    if args.run_agent_one:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from tools_executor import run_tool, parse_action_and_input
        example = eval_ds[0]
        user_msg = example["prompt"][0]["content"]
        assistant_msg = example["completion"][0]["content"]
        actions = parse_action_and_input(assistant_msg)
        print(f"\nAgent sanity check: parsed {len(actions)} actions from first eval example.")
        for name, a in actions[:4]:
            print(f"  {name}: {list(a.keys())}")


if __name__ == "__main__":
    main()
