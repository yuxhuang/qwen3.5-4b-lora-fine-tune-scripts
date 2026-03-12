# Fine-Tuning Plan: Qwen3.5-4B Agentic Loop

This document is the single plan for the project. It includes environment setup (dedicated Miniconda env), training approach, data, and implementation order.

---

## 0. Dedicated Miniconda Environment (Do This First)

Use a **single Miniconda environment** reserved for this training project. Do not reuse the base env or other projects.

### 0.1 Environment name and Python

- **Environment name:** `qwen-agentic-finetune` (or `fine-tune` if you prefer).
- **Python:** 3.12 (match current Miniconda default for compatibility).

### 0.2 Creation and activation

From the project root:

```bash
# Create env (from repo root)
conda create -n qwen-agentic-finetune python=3.12 -y

# Activate before any training or data generation
conda activate qwen-agentic-finetune
```

Or use the project’s `environment.yml` for a reproducible env:

```bash
conda env create -f environment.yml
conda activate qwen-agentic-finetune
```

### 0.3 Install dependencies inside the env

With the env activated:

**Host compatibility:** Install must match your CUDA version. Use the appropriate `--index-url` for your setup (e.g. `cu124` for CUDA 12.4, `cu130` for CUDA 13.x).

```bash
# Use the CUDA wheel matching your GPU. Example for CUDA 13.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Hugging Face stack for LoRA SFT
pip install transformers peft trl datasets accelerate

# Often needed for tokenizers / data
pip install scipy sentencepiece

# Optional: Unsloth (faster training, good for Qwen3.5)
# pip install unsloth
```

Or install from `requirements-train.txt` if present (then reinstall PyTorch with cu130/cu124 as above):

```bash
pip install -r requirements-train.txt
```

### 0.4 Rule: always activate for this project

- All data generation, training, and evaluation for this fine-tuning **must** run with `conda activate qwen-agentic-finetune` (or the chosen name).
- Document this env name in the project README and in any runbooks so others (or you later) use the same env.
- **Training runs on GPU only.** If training fails with a device/CUDA error, see the "If GPU training does not run" steps below.

---

## 1. Goal and Scope

**Target behavior:** The model executes an agentic loop:

1. **Fetch** – Use a tool (e.g. curl) to fetch a URL and get raw HTML.
2. **Save** – Save the fetched HTML to a given local path.
3. **Validate** – Given expected text and a DOM selector, extract content from the saved page and check that the expected text appears at that location.

Inputs: URL, local save path, expected text, DOM/location spec. Output: success/failure of validation (and optionally extracted content).

---

## 2. Tool Definitions

| Tool | Purpose | Key parameters |
|------|--------|----------------|
| `fetch_url` | Fetch HTML from URL | `url` (string) |
| `save_page` | Write HTML to local path | `content` (string), `path` (string) |
| `extract_dom_content` | Extract text at DOM location | `html` or `path`, `selector` (CSS/XPath) |
| `validate_content` | Check expected text in extracted content | `extracted_content`, `expected_text` (string) |

Tools are described in the system/ReAct prompt with **JSON Schema** `parameters`.

---

## 3. Training Data: Source and Format

### 3.1 Data generation

Training data is **generated locally** by `scripts/generate_data.py`. The generator uses **modular page builder functions** with large content pools for combinatorial diversity. Each call to a builder produces a unique (url, html, selector, expected_text) tuple.

Current capabilities:

- **21 page builder categories:** Product, article, dashboard, status, incident, profile, ticket, weather, job, billing, flight, banking, course, tracking, movie, restaurant, podcast, leaderboard, realty, email, thermostat.
- **Large content pools:** 23 company names, 20 person names, 24 products, 21 article titles, 29 domains, and many more — combined for thousands of unique page variations.
- **Multiple extraction targets per page:** Each builder offers 2–5 different (selector, expected_text) pairs, multiplying diversity further.
- **Diverse CSS selectors:** ID (`#product-price`), class (`.article-title`), attribute (`[data-testid="avg-rating"]`), combined selectors.
- **Varied assistant reasoning:** 7–8 phrasings per ReAct step, randomly selected per example.
- **5 user prompt templates:** Different instruction styles (step-by-step, concise, goal-oriented, etc.).
- **Failure cases (~22% of examples)** covering 5 failure modes:
  1. **Validation failure:** Selector matches but extracted text does not contain expected text.
  2. **Selector miss:** CSS selector finds no element (typo, removed element, wrong class).
  3. **Fetch error:** HTTP 404, 500, connection timeout, SSL error, etc.
  4. **Partial/misleading match:** Substring present but semantically wrong (e.g. "Out of Stock" matching "Stock").
  5. **Multiple elements matched:** Selector too broad, returns concatenated text from all matches.

For failure cases, the assistant stops at the failure point, reasons about what went wrong, and gives a clear Final Answer explaining the error.

### 3.2 Format (ReAct for Qwen)

- **Roles:** Only `user` and `assistant` (no function role).
- **User:** Task description + tool list (with JSON Schema).
- **Assistant:** ReAct trace: `Thought:` → `Action:` → `Action Input:` (JSON) → `Observation:` → … → `Final Answer:`.
- Varied phrasing across examples (not a single rigid template).

### 3.3 How to generate data

From the project root with the conda env activated:

```bash
python scripts/generate_data.py --total 800
```

This writes `data/train.jsonl` (~660 examples) and `data/eval.jsonl` (~140 examples). Use `--total`, `--eval-ratio`, and `--seed` to control size, split, and reproducibility.

---

## 4. Dataset Size, Split, and Scaling Strategy

### 4.1 Current baseline

- **Total examples:** 800 (660 train / 140 eval).
- **Composition:** ~78% success cases, ~22% failure cases (5 failure modes).
- **Eval accuracy achieved:** 93.5% (eval loss 0.252, train loss ~0.255, gap ≈ 0).

### 4.2 Target

- **Eval token accuracy:** ~95–97% without overfitting.
- **Key constraint:** Keep the gap between train loss and eval loss small (< 0.03). If the gap widens, the model is overfitting and needs more data, not more epochs.

### 4.3 Scaling strategy

Scale **data quantity and diversity** iteratively until the target accuracy is reached. Do NOT simply add epochs — epochs are already saturated at 5 with the current dataset.

**Iteration plan:**

| Round | Total Examples | Expected Eval Accuracy | Action |
|-------|--------------|----------------------|--------|
| Done  | 300          | 87%                  | Baseline |
| Done  | 800          | 93.5%                | Added 21 builders, content pools |
| Next  | 1500         | ~95%                 | Add 10+ new builders, grow content pools, increase HTML complexity |
| If needed | 2500    | ~96–97%              | More builders, multi-turn examples, messier HTML with noise |

**At each round:**

1. Add new page builder categories (forums, wikis, API docs, changelogs, search results, notifications, calendars, analytics, settings, etc.).
2. Grow content pools for existing builders (more names, prices, titles, etc.).
3. Increase HTML realism: deeper nesting, inline styles, `data-*` attributes, SVG noise, script tags, multiple sibling elements.
4. Regenerate data with `--total <new_size>`.
5. Train with same hyperparameters (epochs=5, lr=1e-5).
6. **Check stopping criteria:**
   - If eval accuracy >= 97%: done.
   - If eval loss - train loss > 0.03: overfitting — add more data, do not add epochs.
   - If eval accuracy plateaus across two rounds: consider adding multi-turn examples or adjusting LoRA rank.

### 4.4 What NOT to do

- **Do not increase epochs beyond 5** as the primary strategy. Epochs 4→5 gained only +0.1pp in the 800-example run.
- **Do not decrease learning rate** hoping for marginal gains — the model converges well at 1e-5.
- **Do not increase LoRA rank** unless data scaling plateaus. Rank 32 is sufficient; higher risks overfitting with limited data.

---

## 5. Model and Training (Summary)

- **Base model:** [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
- **Method:** LoRA (bf16), not full fine-tune.
- **LoRA:** rank 32, alpha 64; target attention + MLP: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Dropout 0.1.
- **Training:** batch size 2, gradient accumulation 8, 5 epochs, eval every epoch.
- **LR:** 1e-5; weight decay 0.01; max length 2048.
- **Stopping criteria:** Stop scaling data when eval token accuracy reaches ~97% with train-eval loss gap < 0.03. Do not add epochs as primary improvement strategy.

See `docs/MACHINE_AND_SETUP.md` for full hyperparameters and machine notes.

### 5.1 Final output directory (`outputs/lora_sft/final/`)

After training completes, `train.py` saves the adapter to `outputs/lora_sft/final/`. This directory contains:

- `adapter_config.json`, `adapter_model.safetensors` — LoRA weights.
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` — tokenizer files.
- `trainer_state.json` — copied from the last checkpoint so training metrics (loss curves, eval accuracy per epoch) are available without the intermediate checkpoints.
- `metadata.json` — written by `train.py` with a UTC timestamp, source checkpoint, final step/epoch, and last eval metrics. This makes it unambiguous which training run produced the final output.

**Important:** Intermediate checkpoints (`checkpoint-*/`) are kept only during training (controlled by `save_total_limit`). They are `.gitignore`d and not committed. Only the `final/` directory and `outputs/gguf/` are version-controlled.

### 5.2 If GPU training does not run

Training **must** run on GPU; the script exits if no usable GPU is found. If training fails:

1. **Verify GPU visibility:** Run `nvidia-smi`; confirm your GPU is listed.
2. **Verify PyTorch + CUDA:** In the project env, run `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`. If `False` or error, reinstall PyTorch with the correct CUDA wheel.
3. **Reinstall PyTorch:** `pip uninstall torch torchvision -y` then `pip install torch torchvision --index-url https://download.pytorch.org/whl/<your-cuda-version>`. Re-run the check in step 2.
4. **If still failing:** Check architecture support for your GPU; consider PyTorch nightly if the stable wheel does not support it. Document the working PyTorch version and CUDA wheel in `docs/MACHINE_AND_SETUP.md`.

Optional: set `ALLOW_CPU_TRAIN=1` to allow CPU training **for debugging only** (not for real training); see `scripts/train.py` and `docs/MACHINE_AND_SETUP.md`.

---

## 6. Project Structure (Target)

```
fine-tune/
├── plan.md                    # This plan
├── environment.yml            # Miniconda env definition
├── requirements-train.txt     # pip deps for training (optional)
├── tools/
│   └── definitions.json       # Tool names + JSON Schema
├── data/
│   ├── train.jsonl
│   └── eval.jsonl
├── scripts/
│   ├── generate_data.py
│   ├── train.py
│   └── run_agent_eval.py
├── config/
│   └── lora_sft.yaml
├── outputs/
│   ├── lora_sft/              # Training checkpoints and final adapter
│   └── gguf/                  # GGUF-converted LoRA adapters (bf16, f16, q8_0)
└── docs/
    └── MACHINE_AND_SETUP.md
```

---

## 7. Convert LoRA Adapter to GGUF

After training, convert the LoRA adapter to GGUF format quantized to **bf16**, **f16**, and **q8_0** for use with llama.cpp-compatible runtimes.

### 7.1 Prerequisites: build llama.cpp (one-time)

The project uses the llama.cpp source tree at `../llama.cpp` (relative to this repo root). No system-wide llama.cpp installation is required — we build once and reference the binaries and scripts in-place.

```bash
cd ../llama.cpp
cmake -B build -DLLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

After the build, the key artifacts are:

| Artifact | Path |
|----------|------|
| **convert_lora_to_gguf.py** | `../llama.cpp/convert_lora_to_gguf.py` |
| **llama-quantize** (if further quantization is needed) | `../llama.cpp/build/bin/llama-quantize` |
| **llama-cli** (for test inference) | `../llama.cpp/build/bin/llama-cli` |

Install the Python GGUF library from the llama.cpp tree so the conversion scripts can import it:

```bash
pip install ../llama.cpp/gguf-py
```

### 7.2 Convert LoRA to GGUF (bf16, f16, q8_0)

Run all three conversions from the project root with the conda env activated. The input is the final adapter directory (`outputs/lora_sft/final`); output GGUFs go into `outputs/gguf/`.

```bash
conda activate qwen-agentic-finetune
mkdir -p outputs/gguf

LLAMA_CPP=../llama.cpp
LORA_DIR=outputs/lora_sft/final

# bf16
python "$LLAMA_CPP/convert_lora_to_gguf.py" \
  --outtype bf16 \
  --outfile outputs/gguf/qwen3.5-4b-agentic-lora-bf16.gguf \
  "$LORA_DIR"

# f16
python "$LLAMA_CPP/convert_lora_to_gguf.py" \
  --outtype f16 \
  --outfile outputs/gguf/qwen3.5-4b-agentic-lora-f16.gguf \
  "$LORA_DIR"

# q8_0
python "$LLAMA_CPP/convert_lora_to_gguf.py" \
  --outtype q8_0 \
  --outfile outputs/gguf/qwen3.5-4b-agentic-lora-q8_0.gguf \
  "$LORA_DIR"
```

The `--base` flag is optional; the script reads the base model ID from `adapter_config.json` and fetches the config from Hugging Face automatically. To skip the network lookup, pass `--base /path/to/Qwen3.5-4B` if you have the base model cached locally.

### 7.3 Verify the outputs

```bash
ls -lh outputs/gguf/
```

Expected: three `.gguf` files. bf16 and f16 will be similar in size (adapter weights only, a few hundred MB); q8_0 will be slightly smaller.

### 7.4 Test inference with the LoRA GGUF (optional)

To smoke-test, load the base model GGUF plus the LoRA adapter in llama-cli:

```bash
../llama.cpp/build/bin/llama-cli \
  -m /path/to/Qwen3.5-4B-base.gguf \
  --lora outputs/gguf/qwen3.5-4b-agentic-lora-q8_0.gguf \
  -p "You are an agentic assistant..." \
  -n 512
```

Replace `/path/to/Qwen3.5-4B-base.gguf` with the actual base model GGUF (download or convert separately with `convert_hf_to_gguf.py`).

---

## 8. Implementation Order

1. **Dedicated Miniconda environment** – Create `qwen-agentic-finetune` (or equivalent), add `environment.yml` and optionally `requirements-train.txt`. Document “always activate this env for this project.”
2. **Tool schema and ReAct template** – Finalize `tools/definitions.json` and one canonical ReAct prompt template.
3. **Data generation** – Implement `scripts/generate_data.py` with modular page builders, content pools, 5 failure modes, varied reasoning, and varied user prompt styles. Start at 800 examples, scale up per Section 4.3.
4. **Training pipeline** – Implement `scripts/train.py` (load base model, LoRA, dataset, chat template; use `config/lora_sft.yaml`).
5. **Agent executor** – Implement runtime that parses Thought/Action/Action Input, runs tools (fetch, save, DOM extract, validate), returns Observation.
6. **Eval script** – Implement `scripts/run_agent_eval.py`; run agent on eval set, compute success rate (and optional metrics).
7. **Iterate on data scale** – After each training run, check eval accuracy and train-eval loss gap. Scale data per Section 4.3 until eval accuracy reaches ~97% with gap < 0.03. Do not add epochs as the primary lever.
8. **Convert LoRA to GGUF** – After a satisfactory training run, build llama.cpp (Section 7.1, one-time), then convert the final adapter to bf16, f16, and q8_0 GGUFs (Section 7.2).

Environment setup is step 1 and is required before any training or data generation.
