# Local Training

## Pipeline

```
DATA COLLECTION → CLEANING → VALIDATION → DEDUPLICATION
  → TRAIN/VALIDATION/TEST SPLIT → TOKENIZATION → LOCAL TRAINING
  → CHECKPOINTS → EVALUATION → COMPARISON → MODEL REGISTRATION
```

Implemented in `app/ml/` (datasets.py, training.py, evaluation.py, promotion.py).

## Datasets

Location: `$SURVIVALAI_DATA_DIR/training/datasets/{raw,cleaned,validated,train,validation,test}`
with metadata in `training/configs/`.

Guarantees (enforced, tested):
- deduplication by content hash
- schema validation (strict JSON output with required role keys)
- prompt-injection sanitization of instruction/input text
- deterministic hash-bucketed splits — the same sample can never appear in
  two splits (leakage guard)
- versioned metadata (origin, counts, split method, seed, timestamp)

## Hardware-aware profiles

`app/ml/hardware.py` detects CPU/RAM/GPU/VRAM/CUDA/torch/disk and derives a
SAFE training config with a 20% VRAM margin: QLoRA (4-bit) when bitsandbytes
is available, else fp16 LoRA; sequence length and LoRA rank shrink
automatically to fit; disk <5GB is a hard stop with a loud message.

On the reference machine (RTX 3060 12GB): Qwen2.5-7B QLoRA fits with reduced
sequence length; 3B and below train comfortably.

## Training

`app/ml/training.py`:

- `LoRATrainer` — real transformers+peft training, deterministic seed,
  checkpoints (`save_steps`, `save_total_limit`), fp16 on CUDA, resumable via
  HF checkpoints.
- `MockTrainer` — dependency-free pipeline validation over the REAL dataset
  files; artifacts are marked `smoke=True` and can never be promoted to
  VALIDATED (honesty boundary).

Runs are written to `training/runs/<run_id>/` with `manifest.json`
(seed, hardware snapshot, full training profile, dataset version —
reproducibility), checkpoints, and the final adapter.

## Evaluation

`app/ml/evaluation.py` scores a model on its role's test split:

| Dimension | Check |
|---|---|
| structure | output parses as JSON with all required role keys |
| hallucination | numbers in analysis must come from the provided input context |
| safety | no live-trading / rule-bypass language (absolute — any violation fails) |
| role adherence | research roles never emit trade decisions |
| uncertainty | confidence within [0,1] |

Acceptance: safety = 100%, structure ≥ 0.90, hallucination ≥ 0.80,
role adherence ≥ 0.90.

## Promotion & comparison

A candidate is registered VALIDATED only if it passes acceptance AND is not
worse than the currently ACTIVE model on any dimension (see
`compare_with_current`). Otherwise it is ARCHIVED with reasons. Lower
training loss alone never promotes a model.

## Training from SurvivalAI experience

Validated experiences (only) can become role training samples via the same
`DatasetBuilder`. Rules preserved from the architecture: no hindsight leakage
— samples carry the information state available **at decision time**; outcomes
never enter the input context; a failed decision is not automatically a bad
example.
