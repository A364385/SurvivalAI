"""End-to-end promotion pipeline and training queue.

Promotion pipeline (enforced, no step skippable):

    BUILD DATASET -> TRAIN -> EVALUATE -> COMPARE AGAINST CURRENT
      -> PASS: REGISTER (VALIDATED)   [activation remains explicit]
      -> FAIL: REGISTER (ARCHIVED/FAILED) with reasons

The TrainingQueue serializes training runs on local hardware (one GPU job at
a time) and exposes queue state for the dashboard.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.ml.datasets import DatasetBuilder
from app.ml.evaluation import EvalResult, HeuristicEvaluator, ModelEvaluator, compare_with_current
from app.ml.model_registry import ModelRecord, ModelRegistry, ModelStatus
from app.ml.training import BaseTrainer, TrainingSpec, build_trainer
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


class PromotionPipeline:
    """Runs the full build->train->evaluate->compare->register flow."""

    def __init__(self, registry: ModelRegistry, trainer: Optional[BaseTrainer] = None,
                 evaluator: Optional[Any] = None,
                 progress: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.registry = registry
        # Lazy trainer: never import torch at construction time (the dashboard
        # and launcher construct pipelines without wanting the ML stack).
        self.trainer = trainer
        # Real evaluator by default (works against any generate_fn); tests may
        # inject the HeuristicEvaluator for GPU-free pipeline validation.
        self.evaluator = evaluator
        self.progress = progress or (lambda event: None)

    # ------------------------------------------------------------------
    def run(
        self,
        role: str,
        dataset_version: str,
        model_version: str,
        base_model: str,
        samples: Optional[List[Dict[str, Any]]] = None,
        spec_overrides: Optional[Dict[str, Any]] = None,
        use_real_trainer: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Execute the pipeline. Returns a full status dict (never raises)."""
        from app.ml.hardware import detect_hardware, recommend_training_profile

        pipeline_id = f"promo_{role}_{model_version}"
        report: Dict[str, Any] = {
            "pipeline_id": pipeline_id,
            "role": role,
            "model_version": model_version,
            "dataset_version": dataset_version,
            "started_at": now_utc().isoformat(),
            "stages": {},
        }
        model_id = f"{role}_{model_version}"
        self.progress({"stage": "start", "role": role, "model_id": model_id})

        try:
            # 1. Dataset ---------------------------------------------------
            self.progress({"stage": "dataset", "status": "building"})
            if samples is None:
                builder = DatasetBuilder(role=role, version=dataset_version)
                dataset_meta = builder.build(self._default_samples(role))
            else:
                builder = DatasetBuilder(role=role, version=dataset_version)
                dataset_meta = builder.build(samples)
            report["stages"]["dataset"] = {"status": "OK", "stats": dataset_meta["stats"]}
            if dataset_meta["stats"]["train"] == 0:
                report["status"] = "FAILED"
                report["error"] = "no valid training samples"
                return report

            # 2. Hardware profile -----------------------------------------
            hardware = detect_hardware()
            profile = recommend_training_profile(hardware, base_model=base_model)
            report["stages"]["hardware"] = {
                "status": "OK",
                "profile": profile.to_dict(),
                "warnings": hardware.warnings,
            }

            # 3. Register model shell (TRAINING) ---------------------------
            record = ModelRecord(
                model_id=model_id,
                role=role,
                base_model=base_model,
                model_version=model_version,
                training_dataset=f"{role}_dataset_{dataset_version}",
                training_date=now_utc().isoformat(),
                training_config=profile.to_dict(),
                quantization="4bit_nf4" if profile.load_in_4bit else "none",
                status=ModelStatus.TRAINING.value,
                artifact_path=None,
                provider="transformers",
            )
            try:
                self.registry.register(record)
            except ValueError:
                # Re-training an existing model_id: reset it to TRAINING.
                self.registry.update_status(model_id, ModelStatus.TRAINING,
                                            notes="re-training")

            # 4. Train -----------------------------------------------------
            self.progress({"stage": "training", "status": "started"})
            spec = TrainingSpec(
                run_id=f"run_{role}_{model_version}_{int(now_utc().timestamp())}",
                role=role,
                model_version=model_version,
                dataset_version=dataset_version,
                base_model=base_model,
                profile=profile,
                seed=42,
                hardware=hardware.to_dict(),
            )
            trainer = self.trainer if self.trainer is not None else build_trainer(use_real=use_real_trainer)
            train_result = trainer.train(spec, progress=self.progress)
            report["stages"]["training"] = {
                "status": train_result.status,
                "steps": train_result.steps,
                "final_loss": train_result.final_loss,
                "adapter_path": train_result.adapter_path,
                "error": train_result.error,
                "smoke": train_result.smoke,
            }
            if train_result.status != "COMPLETED" or not train_result.adapter_path:
                self.registry.update_status(
                    model_id, ModelStatus.FAILED,
                    notes=f"training failed: {train_result.error}",
                )
                report["status"] = "FAILED"
                report["error"] = f"training failed: {train_result.error}"
                return report
            self.registry.update_fields(model_id, artifact_path=train_result.adapter_path)

            # 5. Evaluate --------------------------------------------------
            self.registry.update_status(model_id, ModelStatus.EVALUATING)
            self.progress({"stage": "evaluation", "status": "started"})
            evaluator = self._resolve_evaluator(train_result, use_real_trainer)
            eval_result: EvalResult = evaluator.evaluate(role, dataset_version, model_id)
            self.registry.update_fields(
                model_id, evaluation_results=eval_result.to_dict()
            )
            report["stages"]["evaluation"] = eval_result.to_dict()

            smoke_artifact = bool(getattr(train_result, "smoke", False))

            # 6. Compare against current -----------------------------------
            current = self._current_evaluation(role)
            promote, reasons = compare_with_current(eval_result, current)
            if smoke_artifact:
                promote = False
                reasons.append("smoke artifact cannot be promoted to VALIDATED")

            # 7. Register outcome ------------------------------------------
            if eval_result.passed and promote:
                self.registry.update_status(
                    model_id, ModelStatus.VALIDATED,
                    notes="passed evaluation and comparison",
                )
                report["status"] = "PASSED"
            else:
                self.registry.update_status(
                    model_id, ModelStatus.ARCHIVED,
                    notes="archived: " + "; ".join(reasons or eval_result.failure_reasons),
                )
                report["status"] = "ARCHIVED"
                report["archive_reasons"] = reasons or eval_result.failure_reasons

            report["completed_at"] = now_utc().isoformat()
            self.progress({"stage": "done", "status": report["status"]})
            return report
        except Exception as e:
            logger.exception("Promotion pipeline failed")
            report["status"] = "FAILED"
            report["error"] = f"{type(e).__name__}: {e}"
            try:
                self.registry.update_status(model_id, ModelStatus.FAILED, notes=str(e))
            except Exception:
                pass
            return report

    # ------------------------------------------------------------------
    def _resolve_evaluator(self, train_result: Any, use_real_trainer: Optional[bool]):
        if self.evaluator is not None:
            return self.evaluator
        from app.ml.datasets import DatasetBuilder  # local import: avoids cycle
        # Real trainer path: evaluate the trained adapter via Transformers.
        if not getattr(train_result, "smoke", False):
            from app.services.llm.local_providers import TransformersLocalProvider
            base = getattr(train_result, "base_model", None)
            adapter = train_result.adapter_path
            provider = TransformersLocalProvider(model_id=base or "", adapter_path=adapter)

            def generate_fn(prompt: str) -> str:
                return provider.generate(prompt, max_tokens=700)

            return ModelEvaluator(generate_fn=generate_fn,
                                  dataset_loader=DatasetBuilder(role="x", version="x").load_split)
        return HeuristicEvaluator()

    def _current_evaluation(self, role: str) -> Optional[Dict[str, Any]]:
        for record in self.registry.list_models(role=role):
            if record.status == ModelStatus.ACTIVE.value and record.evaluation_results:
                results = record.evaluation_results
                return {
                    "model_id": record.model_id,
                    "aggregate": results.get("aggregate", 0.0),
                    "scores": results.get("scores", {}),
                }
        return None

    @staticmethod
    def _default_samples(role: str) -> List[Dict[str, Any]]:
        """Minimal deterministic seed samples so the pipeline always has data;
        role datasets built from real experience should replace these."""
        import json as _json
        base_output = {
            "facts": ["seed sample"], "analysis": {"summary": "seed"},
            "impact": {"direction": "neutral"}, "confidence": 0.4,
            "warnings": ["seed sample"], "sources": [],
        }
        samples = []
        for i in range(24):
            samples.append({
                "instruction": f"[{role}] Analyze scenario {i} and produce the structured assessment.",
                "input": f"scenario_id: {i}\nprovided_indicators: price=100.{i} rsi_14=5{i}.0",
                "output": _json.dumps(base_output),
                "meta": {"role": role, "source": "seed"},
            })
        return samples


@dataclass
class QueueItem:
    job_id: str
    role: str
    dataset_version: str
    model_version: str
    base_model: str
    status: str = "QUEUED"       # QUEUED | RUNNING | DONE | FAILED
    result: Optional[Dict[str, Any]] = None
    position: int = 0


class TrainingQueue:
    """Serialized training queue — one GPU job at a time on a local PC."""

    def __init__(self, registry: ModelRegistry, pipeline: Optional[PromotionPipeline] = None):
        self.registry = registry
        self.pipeline = pipeline
        self._items: List[QueueItem] = []
        self._lock = threading.RLock()
        self._worker: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._progress_events: List[Dict[str, Any]] = []

    def enqueue(self, role: str, dataset_version: str, model_version: str,
                base_model: str) -> QueueItem:
        with self._lock:
            item = QueueItem(
                job_id=f"job_{role}_{model_version}_{int(now_utc().timestamp())}",
                role=role, dataset_version=dataset_version,
                model_version=model_version, base_model=base_model,
                position=len(self._items) + 1,
            )
            self._items.append(item)
            if self.pipeline is None:
                self.pipeline = PromotionPipeline(
                    self.registry,
                    progress=self._record_progress,
                )
            self._ensure_worker()
            return item

    def _record_progress(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._progress_events.append({"ts": now_utc().isoformat(), **event})
            self._progress_events = self._progress_events[-200:]

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run_loop, daemon=True,
                                            name="survivalai-training-queue")
            self._worker.start()

    def _run_loop(self) -> None:
        while not self._stop_requested.is_set():
            with self._lock:
                pending = [i for i in self._items if i.status == "QUEUED"]
            if not pending:
                break
            item = pending[0]
            with self._lock:
                item.status = "RUNNING"
            try:
                result = self.pipeline.run(
                    role=item.role,
                    dataset_version=item.dataset_version,
                    model_version=item.model_version,
                    base_model=item.base_model,
                )
                with self._lock:
                    item.result = result
                    item.status = "DONE" if result.get("status") in ("PASSED", "ARCHIVED") else "FAILED"
            except Exception as e:
                logger.exception("Queue job failed")
                with self._lock:
                    item.status = "FAILED"
                    item.result = {"error": str(e)}

    def state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "items": [
                    {"job_id": i.job_id, "role": i.role, "status": i.status,
                     "model_version": i.model_version,
                     "result": (i.result or {}).get("status") if i.result else None}
                    for i in self._items
                ],
                "progress": list(self._progress_events[-30:]),
                "running": any(i.status == "RUNNING" for i in self._items),
            }

    def stop(self) -> None:
        self._stop_requested.set()
