"""Versioned local dataset pipeline.

Directory layout (under SURVIVALAI_DATA_DIR or ./data/training):

training/
  datasets/raw/         JSONL as collected
  datasets/cleaned/     deduplicated + sanitized
  datasets/validated/   schema-validated samples
  datasets/train|validation|test/   final splits
  configs/ runs/ checkpoints/ evaluations/ exports/

Samples are instruction-format dicts:
  {"instruction": str, "input": str, "output": str (strict JSON string),
   "meta": {"role": ..., "source": ..., "created_at": ..., "as_of": ...}}

Guarantees:
- dedup by (instruction, output) hash
- validation: output must parse as JSON and contain required role keys
- split BEFORE any training; test set never overlaps train (hash-bucketed by
  content hash so the same sample can never leak across splits)
- every dataset version writes a metadata JSON (origin, counts, hashes, date)
"""

import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.services.llm.prompt_defense import sanitize_untrusted
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)

REQUIRED_OUTPUT_KEYS = {"facts", "analysis", "impact", "confidence", "warnings"}


def data_root() -> Path:
    root = os.environ.get("SURVIVALAI_DATA_DIR")
    base = Path(root) if root else Path("data")
    return base / "training"


@dataclass
class DatasetStats:
    total: int = 0
    duplicates_removed: int = 0
    invalid_removed: int = 0
    sanitized_count: int = 0
    train: int = 0
    validation: int = 0
    test: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sample_hash(sample: Dict[str, Any]) -> str:
    basis = json.dumps(
        [sample.get("instruction", ""), sample.get("input", ""), sample.get("output", "")],
        sort_keys=True,
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def validate_sample(sample: Dict[str, Any], role: Optional[str] = None) -> Tuple[bool, str]:
    """A sample is valid iff output parses as JSON and has required keys."""
    if not isinstance(sample, dict):
        return False, "not a dict"
    for key in ("instruction", "output"):
        if not isinstance(sample.get(key), str) or not sample[key].strip():
            return False, f"missing or empty field: {key}"
    try:
        output = json.loads(sample["output"])
    except json.JSONDecodeError as e:
        return False, f"output not valid JSON: {e}"
    if not isinstance(output, dict):
        return False, "output JSON is not an object"
    missing = REQUIRED_OUTPUT_KEYS - set(output.keys())
    if missing:
        return False, f"output missing keys: {sorted(missing)}"
    confidence = output.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        return False, "confidence must be a number in [0,1]"
    if role:
        meta_role = (sample.get("meta") or {}).get("role")
        if meta_role and meta_role != role:
            return False, f"role mismatch: {meta_role} != {role}"
    return True, "ok"


class DatasetBuilder:
    """Builds, cleans, validates, splits and exports a role dataset version."""

    def __init__(self, role: str, version: str, seed: int = 42):
        self.role = role
        self.version = version
        self.seed = seed
        self.root = data_root()
        self.dataset_id = f"{role}_dataset_{version}"

    # ------------------------------------------------------------------
    def _dir(self, stage: str) -> Path:
        d = self.root / "datasets" / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    def build(
        self,
        samples: List[Dict[str, Any]],
        source: str = "synthetic",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> Dict[str, Any]:
        """Run the full pipeline: clean -> dedup -> validate -> split -> export."""
        stats = DatasetStats(total=len(samples))
        rng = random.Random(self.seed)

        cleaned: List[Dict[str, Any]] = []
        seen_hashes = set()
        for raw in samples:
            if not isinstance(raw, dict):
                stats.invalid_removed += 1
                continue
            sample = dict(raw)
            # Sanitize untrusted instruction/input text (prompt-injection defense
            # applies to training data too).
            for key in ("instruction", "input"):
                original = sample.get(key, "")
                sanitized = sanitize_untrusted(str(original))
                if sanitized != original:
                    stats.sanitized_count += 1
                sample[key] = sanitized
            sample.setdefault("meta", {})
            sample["meta"].setdefault("role", self.role)
            sample["meta"].setdefault("source", source)
            sample["meta"].setdefault("created_at", now_utc().isoformat())

            h = _sample_hash(sample)
            if h in seen_hashes:
                stats.duplicates_removed += 1
                continue
            seen_hashes.add(h)

            ok, reason = validate_sample(sample, self.role)
            if not ok:
                stats.invalid_removed += 1
                logger.debug("Rejected sample: %s", reason)
                continue
            cleaned.append(sample)

        # Save cleaned + validated copies.
        cleaned_path = self._dir("cleaned") / f"{self.dataset_id}.jsonl"
        self._write_jsonl(cleaned_path, cleaned)
        validated_path = self._dir("validated") / f"{self.dataset_id}.jsonl"
        self._write_jsonl(validated_path, cleaned)

        # Deterministic split: hash-bucket by sample hash so identical content
        # can never appear in two splits (leakage guard), independent of order.
        train, validation, test = [], [], []
        h_train = int(split_ratios[0] * 100)
        h_val = h_train + int(split_ratios[1] * 100)
        for sample in cleaned:
            bucket = int(_sample_hash(sample)[:8], 16) % 100
            if bucket < h_train:
                train.append(sample)
            elif bucket < h_val:
                validation.append(sample)
            else:
                test.append(sample)
        stats.train, stats.validation, stats.test = len(train), len(validation), len(test)

        self._write_jsonl(self._dir("train") / f"{self.dataset_id}.jsonl", train)
        self._write_jsonl(self._dir("validation") / f"{self.dataset_id}.jsonl", validation)
        self._write_jsonl(self._dir("test") / f"{self.dataset_id}.jsonl", test)

        metadata = {
            "dataset_id": self.dataset_id,
            "role": self.role,
            "version": self.version,
            "created_at": now_utc().isoformat(),
            "source": source,
            "seed": self.seed,
            "split_ratios": list(split_ratios),
            "split_method": "sha256_bucket",
            "leakage_check": "hash_bucketed_exact_dedup",
            "stats": stats.to_dict(),
            "files": {
                "cleaned": str(cleaned_path),
                "validated": str(validated_path),
                "train": str(self._dir("train") / f"{self.dataset_id}.jsonl"),
                "validation": str(self._dir("validation") / f"{self.dataset_id}.jsonl"),
                "test": str(self._dir("test") / f"{self.dataset_id}.jsonl"),
            },
        }
        meta_path = self.root / "configs" / f"{self.dataset_id}_metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Dataset %s built: %s", self.dataset_id, stats.to_dict())
        return metadata

    def _write_jsonl(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    def load_split(self, split: str) -> List[Dict[str, Any]]:
        path = self._dir(split) / f"{self.dataset_id}.jsonl"
        if not path.exists():
            return []
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


def load_dataset_metadata(role: str, version: str) -> Optional[Dict[str, Any]]:
    dataset_id = f"{role}_dataset_{version}"
    meta_path = data_root() / "configs" / f"{dataset_id}_metadata.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))
