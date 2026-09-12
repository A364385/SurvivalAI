"""Hardware-aware resource manager.

Detects CPU/RAM/GPU/VRAM/disk/Python/ML-library capabilities at runtime and
derives SAFE training configurations (never blindly allocating everything):
batch size, gradient accumulation, sequence length, LoRA rank, quantization
mode, and estimated VRAM/RAM usage. Includes safety margins and loud warnings
about low disk space.

This module works without torch/psutil installed: it degrades to conservative
CPU-only recommendations so the rest of the system never breaks.
"""

import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HardwareProfile:
    """Detected hardware/software capabilities."""
    platform: str = platform.platform()
    python_version: str = sys.version.split()[0]
    cpu_name: str = "unknown"
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_gb: float = 0.0
    gpu_name: Optional[str] = None
    gpu_vram_gb: float = 0.0
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    torch_version: Optional[str] = None
    transformers_version: Optional[str] = None
    peft_version: Optional[str] = None
    datasets_version: Optional[str] = None
    bitsandbytes_available: bool = False
    disk_free_gb: float = 0.0
    disk_path: str = "."
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingProfile:
    """A safe training configuration derived from hardware."""
    method: str                 # "lora" | "qlora" | "full_ft_unavailable"
    base_model: str
    lora_rank: int
    lora_alpha: int
    batch_size: int
    gradient_accumulation: int
    sequence_length: int
    learning_rate: float
    epochs: int
    load_in_4bit: bool
    estimated_vram_gb: float
    estimated_ram_gb: float
    recommended_num_workers: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Conservative base-model ladder for consumer GPUs (approx. training VRAM
# including optimizer states for LoRA at seq 1024, batch 1 + inference overhead).
def candidate_base_models(vram_gb: float) -> List[str]:
    if vram_gb >= 22:
        return ["Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
    if vram_gb >= 10:
        # 7B QLoRA trains comfortably in ~10-11GB; LoRA fp16 needs more.
        return ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
    if vram_gb >= 6:
        return ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
    return ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]


def detect_hardware(probe_path: str = ".") -> HardwareProfile:
    """Probe the machine; never raises (missing libs degrade gracefully)."""
    profile = HardwareProfile(disk_path=str(probe_path))
    warnings: List[str] = []

    try:
        import psutil
        vm = psutil.virtual_memory()
        profile.ram_gb = round(vm.total / 1e9, 1)
        profile.cpu_threads = psutil.cpu_count(logical=True) or 0
        profile.cpu_cores = psutil.cpu_count(logical=False) or profile.cpu_threads
        du = psutil.disk_usage(probe_path)
        profile.disk_free_gb = round(du.free / 1e9, 1)
    except Exception as e:
        warnings.append(f"psutil probe failed: {e}")

    try:
        import torch
        profile.torch_version = torch.__version__
        profile.cuda_available = bool(torch.cuda.is_available())
        if profile.cuda_available:
            props = torch.cuda.get_device_properties(0)
            profile.gpu_name = props.name
            profile.gpu_vram_gb = round(props.total_memory / 1e9, 1)
            major, minor = torch.cuda.get_device_capability(0)
            profile.cuda_version = f"{major}.{minor}"
    except Exception as e:
        warnings.append(f"torch probe failed: {e}")

    try:
        import transformers
        profile.transformers_version = transformers.__version__
    except Exception:
        pass
    try:
        import peft
        profile.peft_version = peft.__version__
    except Exception:
        pass
    try:
        import datasets
        profile.datasets_version = datasets.__version__
    except Exception:
        pass
    try:
        import bitsandbytes  # noqa: F401
        profile.bitsandbytes_available = True
    except Exception:
        profile.bitsandbytes_available = False
        if profile.cuda_available:
            warnings.append(
                "bitsandbytes not available — QLoRA disabled; will use fp16 LoRA"
            )

    if profile.disk_free_gb < 5:
        warnings.append(
            f"Only {profile.disk_free_gb}GB free on {profile.disk_path} — model "
            "downloads and checkpoints will fail. Free space or set "
            "SURVIVALAI_DATA_DIR to another drive before training."
        )
    if profile.cuda_available and profile.gpu_vram_gb < 6:
        warnings.append(
            f"GPU VRAM ({profile.gpu_vram_gb}GB) is small; expect 0.5B-1.5B models only"
        )

    profile.warnings = warnings
    logger.info(
        "Hardware: cpu=%s ram=%sGB gpu=%s vram=%sGB cuda=%s disk_free=%sGB",
        profile.cpu_name or profile.cpu_threads, profile.ram_gb, profile.gpu_name,
        profile.gpu_vram_gb, profile.cuda_available, profile.disk_free_gb,
    )
    return profile


def recommend_training_profile(
    hardware: HardwareProfile,
    base_model: Optional[str] = None,
    epochs: int = 3,
) -> TrainingProfile:
    """Derive a SAFE training config with margins. Prefers QLoRA when
    bitsandbytes is available, else fp16 LoRA; shrinks parameters until the
    estimated VRAM fits with a 20% safety margin."""
    notes: List[str] = []
    vram = hardware.gpu_vram_gb if hardware.cuda_available else 0.0

    candidates = candidate_base_models(vram)
    if base_model:
        candidates = [base_model] + [c for c in candidates if c != base_model]
    chosen_base = candidates[0]

    load_in_4bit = hardware.cuda_available and hardware.bitsandbytes_available
    method = "qlora" if load_in_4bit else ("lora" if hardware.cuda_available else "cpu_lora")

    # Parameter count estimate from model name (billions).
    def param_b(model_name: str) -> float:
        for token in model_name.split("-"):
            token = token.strip().lower()
            for suffix, mult in (("b", 1.0), ("0.5b", 0.5), ("1.5b", 1.5), ("3b", 3.0),
                                 ("7b", 7.0), ("14b", 14.0), ("72b", 72.0)):
                if token.endswith(suffix) and token[:-len(suffix)].replace(".", "").isdigit() or token == suffix:
                    try:
                        return float(token.rstrip("b")) if token.endswith("b") else 0.5
                    except ValueError:
                        continue
        return 7.0

    # VRAM model (GB): weights (4bit ~0.55 bytes/param, fp16 ~2.2) + LoRA
    # optimizer + activations at chosen seq len, batch 1 + CUDA overhead.
    seq_len = 2048
    batch = 1
    grad_accum = 16
    lora_rank = 16

    def estimate_vram_gb(p_b: float, four_bit: bool, seq: int, rank: int) -> float:
        weights = p_b * (0.55 if four_bit else 2.2)
        lora_overhead = 0.8 + rank / 64.0
        activations = 0.35 * (seq / 1024.0)
        return weights + lora_overhead + activations + 1.2

    p_b = param_b(chosen_base)
    est = estimate_vram_gb(p_b, load_in_4bit, seq_len, lora_rank)
    usable = vram * 0.8 if vram > 0 else 0.0

    # Shrink loop: sequence length -> rank -> model (handled by caller retry).
    while vram > 0 and est > usable and seq_len > 512:
        seq_len = max(512, seq_len // 2)
        notes.append(f"reduced sequence length to {seq_len} to fit VRAM")
        est = estimate_vram_gb(p_b, load_in_4bit, seq_len, lora_rank)
    while vram > 0 and est > usable and lora_rank > 4:
        lora_rank = max(4, lora_rank // 2)
        notes.append(f"reduced LoRA rank to {lora_rank} to fit VRAM")
        est = estimate_vram_gb(p_b, load_in_4bit, seq_len, lora_rank)

    if vram == 0:
        notes.append("No CUDA GPU detected: CPU LoRA training will be very slow; "
                     "prefer LM Studio/Ollama for inference and tiny datasets for training")
        est = 0.0

    est_ram = max(4.0, p_b * (0.8 if load_in_4bit else 2.4) + 2.0)
    if hardware.ram_gb > 0 and est_ram > hardware.ram_gb * 0.8:
        notes.append("RAM estimate is close to system limit; close other applications")

    lr = 2e-4 if method == "qlora" else 1e-4
    workers = max(1, min(4, (hardware.cpu_cores or 4) // 4))

    return TrainingProfile(
        method=method,
        base_model=chosen_base,
        lora_rank=lora_rank,
        lora_alpha=lora_rank * 2,
        batch_size=batch,
        gradient_accumulation=grad_accum,
        sequence_length=seq_len,
        learning_rate=lr,
        epochs=epochs,
        load_in_4bit=load_in_4bit,
        estimated_vram_gb=round(est, 2),
        estimated_ram_gb=round(est_ram, 2),
        recommended_num_workers=workers,
        notes=notes,
    )
