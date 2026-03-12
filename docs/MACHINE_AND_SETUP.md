# Machine Requirements & Recommended Fine-Tuning Setup

## 1. Minimum Machine Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **GPU** | NVIDIA GPU with ≥24 GB VRAM (Ampere or newer) | ≥48 GB VRAM |
| **CUDA** | 12.4+ | Match your GPU architecture |
| **CPU** | 8+ cores | 32+ cores |
| **RAM** | 32 GB | 64+ GB |
| **Disk** | 50 GB free | 100+ GB free |
| **Python** | 3.12 (miniconda) | 3.12 |

**Note:** Use the **dedicated Miniconda environment** for this project (see project `plan.md`). Do not use the base env. The env name is `qwen-agentic-finetune`; create it with `conda env create -f environment.yml` from the project root. The base model for fine-tuning is **[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)**.

---

## 2. Recommendation: Use LoRA (Not Full Fine-Tune)

With the current dataset size (800–2500 examples, including ~22% failure cases), full fine-tuning would likely **overfit**. LoRA is the better choice:

- **Fewer trainable parameters** → more robust with limited datasets.
- **Base model stays frozen** → general Qwen abilities (reasoning, formatting) are preserved.
- LoRA only needs ~12–20 GB VRAM; full fine-tune risks worse validation and unstable training at this data scale.

So: **best setup = LoRA (bf16)** with conservative hyperparameters and regularization.

---

## 3. Recommended Training Setup

### 3.1 Method and precision

- **Method:** LoRA (PEFT), not QLoRA and not full fine-tune.
- **Precision:** bf16 (full LoRA; no 4-bit).
- **Framework:** Hugging Face PEFT + TRL `SFTTrainer`, or Unsloth if you prefer (both support Qwen3.5-4B).

### 3.2 LoRA config

- **Rank (r):** 32 (avoid 64+ to reduce overfitting).
- **Alpha:** 64 (`alpha = 2 * r`).
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (attention + MLP for more capacity with diverse data).
- **Dropout:** 0.1 for LoRA layers.

### 3.3 Training hyperparameters (small dataset)

- **Batch size:** 2–4 per device (keep small for stability and generalization).
- **Gradient accumulation:** 4–8 steps → effective batch size 16–32.
- **Epochs:** 5 (do not increase as primary improvement lever; see Section 4 note below).
- **Learning rate:** 1e-5 (stable across dataset sizes up to 2500).
- **LR scheduler:** Cosine or linear warmup (e.g. 10% warmup steps).
- **Weight decay:** 0.01–0.02 (helps with overfitting on tiny data).
- **Max sequence length:** 1024 or 2048 (enough for ReAct + short HTML snippets).
- **Gradient checkpointing:** Optional; enable to trade compute for lower peak VRAM if you want larger batch/longer context.

### 3.4 Data and validation

- **Train/eval split:** ~83/17 split. Current: 800 total (660 train / 140 eval). ~22% failure cases.
- **Validation:** Eval every epoch. Monitor train-eval loss gap; if gap > 0.03, scale data instead of adding epochs.
- **Target:** Scale data until eval token accuracy reaches ~97% with train-eval loss gap < 0.03.

### 3.5 Expected resource use (LoRA, 4B)

- **VRAM:** Roughly 12–20 GB (bf16 + batch 2–4, max length 2048).
- **Training time:** On the order of minutes to ~15 minutes for 5 epochs over 250 examples.

---

## 4. Scaling Strategy and What Not to Do

### Scaling strategy (see plan.md Section 4.3 for full details)

Scale **data quantity and diversity** iteratively. Do NOT add epochs as the primary improvement lever.

| Round | Total Examples | Eval Accuracy | Status |
|-------|--------------|---------------|--------|
| v1    | 300          | 87%           | Done   |
| v2    | 800          | 93.5%         | Done   |
| v3    | 1500         | ~95%          | Next   |
| v4    | 2500         | ~96–97%       | If needed |

**Stopping criteria:** eval token accuracy ≥ 97% with train-eval loss gap < 0.03.

### What not to do

- **Do not full fine-tune** – overfitting risk is high; LoRA is the right choice.
- **Do not use a huge LoRA rank** (e.g. 64 or 128) – keep 32 unless data scaling plateaus.
- **Do not add epochs beyond 5** – epochs 4→5 gained only +0.1pp in the 800-example run. More data is the lever.
- **Do not use a large batch size** – 2 per device is enough; use gradient accumulation for effective batch size.

---

## 5. Dedicated Miniconda Environment (Required)

This project uses a **single Miniconda environment** reserved for this training. See **`plan.md`** in the project root for the full environment section.

**Quick setup from project root:**

```bash
conda env create -f environment.yml
conda activate qwen-agentic-finetune
```

Then install PyTorch with CUDA matching your GPU. Use the appropriate `--index-url` for your CUDA version (e.g. `cu124` for CUDA 12.4, `cu130` for CUDA 13.x):

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**Compatibility check** (after creating the env and installing PyTorch):

```bash
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Confirm your GPU is listed and PyTorch sees it.

Optional: **Unsloth** (faster training, good for Qwen3.5):

```bash
pip install unsloth
```

---

### 5.1 If GPU training does not run

Training runs on GPU only. If the training script exits with a device/CUDA error:

1. **Verify GPU visibility:** Run `nvidia-smi`; confirm your GPU is listed and no error.
2. **Verify PyTorch + CUDA:** Run `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` in the project env. If `False` or error, reinstall PyTorch with the correct CUDA wheel.
3. **Reinstall PyTorch:** `pip uninstall torch torchvision -y` then `pip install torch torchvision --index-url https://download.pytorch.org/whl/<your-cuda-version>`. Re-run the check in step 2.
4. **If still failing:** Check architecture support for your GPU; consider PyTorch nightly if the stable wheel does not support it. Document the working combination (PyTorch version + CUDA wheel) here.

Optional: set `ALLOW_CPU_TRAIN=1` to allow CPU training **for debugging only** (not for real training).

---

## 6. Summary Table

| Choice | Recommendation |
|--------|----------------|
| **Method** | LoRA (bf16) |
| **LoRA rank** | 32 |
| **LoRA targets** | attention + MLP (q/k/v/o/gate/up/down_proj) |
| **LoRA dropout** | 0.1 |
| **Batch size** | 2 |
| **Gradient accumulation** | 8 |
| **Epochs** | 5 + early stopping |
| **Learning rate** | 1e-5 |
| **Weight decay** | 0.01 |
| **Max length** | 2048 |
| **Train/eval split** | ~83/17 split (start 800, scale to 2500; ~22% failure cases) |

This setup targets an iteratively scaled dataset (800→2500 examples). The primary improvement lever is data quantity and diversity, not epochs. Target: eval token accuracy ~97% with train-eval loss gap < 0.03.
