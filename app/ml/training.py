"""Local training runner for role-specialized LoRA/QLoRA adapters.

Two execution paths:

1. `LoRATrainer` — real training via transformers+peft (CUDA, fp16/4-bit).
   Used when torch/transformers/peft are installed; supports checkpoints and
   deterministic seeds.

2. `MockTrainer` — dependency-free smoke-test path used by CI/tests and by the
   "verify the pipeline works" flow. It produces a REAL adapter artifact
   directory with config metadata but no fine-tuned weights, and clearly marks
   itself as a smoke artifact (status can never become VALIDATED via this
   path unless real weights exist — see training_run.metadata["smoke"]).

Both paths share TrainingSpec/TrainingResult and write the same on-disk
structure under training/runs/<run_id>/.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.ml.datasets import data_root, load_dataset_metadata
from app.ml.hardware import HardwareProfile, TrainingProfile
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


@dataclass
class TrainingSpec:
    run_id: str
    role: str
    model_version: str
    dataset_version: str
    base_model: str
    profile: TrainingProfile
    seed: int = 42
    hardware: Optional[Dict[str, Any]] = None


@dataclass
class TrainingResult:
    run_id: str
    status: str                      # COMPLETED | FAILED
    adapter_path: Optional[str] = None
    final_loss: Optional[float] = None
    steps: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    smoke: bool = False
    loss_history: List[float] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)


class BaseTrainer:
    """Shared run-directory plumbing."""

    def __init__(self, resources: Optional[Dict[str, Callable[[], Dict[str, Any]]]] = None):
        self.resources = resources or {}

    def _run_dir(self, run_id: str) -> Path:
        d = data_root() / "runs" / run_id
        (d / "checkpoints").mkdir(parents=True, exist_ok=True)
        return d

    def _write_manifest(self, run_id: str, spec: TrainingSpec, extra: Dict[str, Any]) -> None:
        manifest = {
            "run_id": run_id,
            "created_at": now_utc().isoformat(),
            "spec": {
                "role": spec.role,
                "model_version": spec.model_version,
                "dataset_version": spec.dataset_version,
                "base_model": spec.base_model,
                "seed": spec.seed,
                "training_profile": spec.profile.to_dict(),
                "hardware": spec.hardware or {},
            },
            **extra,
        }
        (self._run_dir(run_id) / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )


class MockTrainer(BaseTrainer):
    """Deterministic, dependency-free trainer used for pipeline smoke tests.

    It performs real dataset iteration, computes a deterministic pseudo-loss
    from the data (so different datasets produce different runs), writes
    checkpoints and a final adapter directory. The artifact is explicitly a
    smoke artifact: `smoke=True` prevents it from being presented as a real
    fine-tune (evaluation/registration reject smoke artifacts for VALIDATED).
    """

    def train(self, spec: TrainingSpec, max_steps: int = 20,
              progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> TrainingResult:
        started = time.time()
        result = TrainingResult(run_id=spec.run_id, status="FAILED", smoke=True)
        try:
            self._write_manifest(spec.run_id, spec, {"trainer": "mock", "smoke": True})
            dataset = load_dataset_metadata(spec.role, spec.dataset_version)
            if dataset is None:
                result.error = f"dataset {spec.role}_{spec.dataset_version} not found"
                return result
            train_file = Path(dataset["files"]["train"])
            if not train_file.exists() or train_file.stat().st_size == 0:
                result.error = "train split is empty"
                return result

            rows = [json.loads(l) for l in train_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not rows:
                result.error = "train split has no valid rows"
                return result

            loss_history: List[float] = []
            steps = 0
            for step in range(max_steps):
                row = rows[step % len(rows)]
                # Deterministic pseudo-loss derived from content (real data flow,
                # no randomness needed).
                text = row.get("output", "")
                pseudo_loss = max(0.05, 2.0 - (len(text) % 97) / 50.0 - step * 0.05)
                loss_history.append(round(pseudo_loss, 4))
                steps = step + 1
                if step % 5 == 0 or step == max_steps - 1:
                    ckpt = self._run_dir(spec.run_id) / "checkpoints" / f"step_{steps}.json"
                    ckpt.write_text(json.dumps({
                        "step": steps, "loss": pseudo_loss, "ts": now_utc().isoformat(),
                    }), encoding="utf-8")
                    result.checkpoints.append(str(ckpt))
                if progress:
                    progress({"step": steps, "loss": pseudo_loss, "max_steps": max_steps})

            adapter_dir = self._run_dir(spec.run_id) / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapter_config.json").write_text(json.dumps({
                "role": spec.role,
                "base_model": spec.base_model,
                "smoke": True,
                "lora": spec.profile.to_dict(),
                "seed": spec.seed,
                "created_at": now_utc().isoformat(),
            }, indent=2), encoding="utf-8")

            result.status = "COMPLETED"
            result.adapter_path = str(adapter_dir)
            result.final_loss = loss_history[-1] if loss_history else None
            result.steps = steps
            result.loss_history = loss_history
            result.duration_seconds = round(time.time() - started, 2)
            (self._run_dir(spec.run_id) / "result.json").write_text(
                json.dumps(asdict(result), indent=2), encoding="utf-8"
            )
            logger.info("Mock training completed: run=%s steps=%s loss=%s",
                        spec.run_id, steps, result.final_loss)
            return result
        except Exception as e:
            logger.exception("Mock training failed")
            result.error = str(e)
            result.duration_seconds = round(time.time() - started, 2)
            (self._run_dir(spec.run_id) / "result.json").write_text(
                json.dumps(asdict(result), indent=2), encoding="utf-8"
            )
            return result


class LoRATrainer(BaseTrainer):
    """Real LoRA/QLoRA trainer (transformers + peft). Imports lazily so the
    rest of the system works without the ML stack installed."""

    def train(self, spec: TrainingSpec, max_steps: int = 200,
              progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> TrainingResult:
        started = time.time()
        result = TrainingResult(run_id=spec.run_id, status="FAILED")
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig, get_peft_model
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                Trainer, TrainingArguments,
            )
        except ImportError as e:
            result.error = (
                f"ML stack unavailable ({e}); install torch/transformers/peft/datasets "
                "or use the mock trainer for pipeline smoke tests"
            )
            self._write_manifest(spec.run_id, spec, {"trainer": "lora", "error": result.error})
            return result

        try:
            self._write_manifest(spec.run_id, spec, {"trainer": "lora"})
            torch.manual_seed(spec.seed)

            dataset_meta = load_dataset_metadata(spec.role, spec.dataset_version)
            if dataset_meta is None:
                result.error = f"dataset {spec.role}_{spec.dataset_version} not found"
                return result
            train_file = dataset_meta["files"]["train"]
            rows = [json.loads(l) for l in Path(train_file).read_text(encoding="utf-8").splitlines() if l.strip()]
            if not rows:
                result.error = "train split is empty"
                return result

            tokenizer = AutoTokenizer.from_pretrained(spec.base_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            def format_row(row: Dict[str, Any]) -> str:
                return (
                    f"### Instruction:\n{row['instruction']}\n\n"
                    f"### Input:\n{row.get('input', '')}\n\n"
                    f"### Response:\n{row['output']}"
                )

            ds = Dataset.from_list([{"text": format_row(r)} for r in rows])

            def tokenize(batch):
                return tokenizer(
                    batch["text"],
                    truncation=True,
                    max_length=spec.profile.sequence_length,
                    padding="max_length" if spec.profile.batch_size > 1 else False,
                )

            tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

            load_kwargs: Dict[str, Any] = {"torch_dtype": torch.float16}
            if spec.profile.load_in_4bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs["device_map"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(spec.base_model, **load_kwargs)
            if not spec.profile.load_in_4bit and torch.cuda.is_available():
                model = model.to("cuda")

            lora_config = LoraConfig(
                r=spec.profile.lora_rank,
                lora_alpha=spec.profile.lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            use_fp16 = torch.cuda.is_available()
            args = TrainingArguments(
                output_dir=str(self._run_dir(spec.run_id) / "checkpoints"),
                per_device_train_batch_size=spec.profile.batch_size,
                gradient_accumulation_steps=spec.profile.gradient_accumulation,
                num_train_epochs=spec.profile.epochs,
                max_steps=max_steps if max_steps > 0 else -1,
                learning_rate=spec.profile.learning_rate,
                logging_steps=5,
                save_steps=50,
                save_total_limit=3,
                fp16=use_fp16,
                bf16=False,
                report_to=[],
                seed=spec.seed,
                dataloader_num_workers=spec.profile.recommended_num_workers,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=tokenized,
            )
            trainer.train(resume_from_checkpoint=None)

            adapter_dir = self._run_dir(spec.run_id) / "adapter"
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))

            result.status = "COMPLETED"
            result.adapter_path = str(adapter_dir)
            logs = [h for h in trainer.state.log_history if "loss" in h]
            result.final_loss = logs[-1]["loss"] if logs else None
            result.loss_history = [round(h["loss"], 4) for h in logs]
            result.steps = trainer.state.global_step
            result.duration_seconds = round(time.time() - started, 2)
            result.checkpoints = sorted(
                str(p) for p in (self._run_dir(spec.run_id) / "checkpoints").glob("checkpoint-*")
            )
            (self._run_dir(spec.run_id) / "result.json").write_text(
                json.dumps(asdict(result), indent=2), encoding="utf-8"
            )
            logger.info("LoRA training completed: run=%s steps=%s loss=%s",
                        spec.run_id, result.steps, result.final_loss)
            return result
        except Exception as e:
            logger.exception("LoRA training failed")
            result.error = f"{type(e).__name__}: {e}"
            result.duration_seconds = round(time.time() - started, 2)
            (self._run_dir(spec.run_id) / "result.json").write_text(
                json.dumps(asdict(result), indent=2), encoding="utf-8"
            )
            return result


def build_trainer(use_real: Optional[bool] = None) -> BaseTrainer:
    """Choose the trainer: real if the ML stack is present AND importable
    (a broken install, e.g. DLL init failure, counts as unavailable), else
    the deterministic mock trainer."""
    if use_real is None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            import peft  # noqa: F401
            use_real = True
        except Exception:  # ImportError AND runtime DLL/load failures
            logger.warning("ML stack unavailable or broken; using mock trainer")
            use_real = False
    return LoRATrainer() if use_real else MockTrainer()
