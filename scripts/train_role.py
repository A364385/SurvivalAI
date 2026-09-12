#!/usr/bin/env python
"""SurvivalAI one-command local training CLI.

Usage:
    python scripts/train_role.py --role market_research --version v0.1
    python scripts/train_role.py --role market_research --version v0.1 --dataset my_v2
    python scripts/train_role.py --role market_research --smoke   (no GPU/ML stack needed)

Runs the full promotion pipeline:
    validate hardware -> build/validate dataset -> train (LoRA/QLoRA) ->
    evaluate -> compare against current model -> register (or archive)

A model is NEVER auto-activated; activation happens in the dashboard
(Models page) or via --activate (only for VALIDATED models).
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SurvivalAI role model")
    parser.add_argument("--role", required=True,
                        help="role id, e.g. market_research, news_research, ceo")
    parser.add_argument("--version", default="v0.1", help="model version tag")
    parser.add_argument("--dataset", default="v1", help="dataset version to train on")
    parser.add_argument("--base-model", default=None,
                        help="HF base model id (auto-selected from hardware if omitted)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--smoke", action="store_true",
                        help="pipeline smoke test: real files, no GPU/ML stack needed")
    parser.add_argument("--force-mock", action="store_true",
                        help="force the deterministic mock trainer")
    parser.add_argument("--activate", action="store_true",
                        help="activate after successful validation (explicit opt-in)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--list-roles", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.data_dir:
        os.environ["SURVIVALAI_DATA_DIR"] = args.data_dir

    from app.ml.roles import all_roles, get_role
    if args.list_roles:
        for role in all_roles():
            print(f"{role:22s} {get_role(role).display_name}")
        return 0
    if args.role not in all_roles():
        print(f"ERROR: unknown role '{args.role}'. Use --list-roles.")
        return 2

    from app.ml.model_registry import ModelRegistry, ModelStatus
    from app.ml.promotion import PromotionPipeline
    from app.ml.hardware import detect_hardware, recommend_training_profile

    data_dir = Path(os.environ.get("SURVIVALAI_DATA_DIR", "data"))
    registry = ModelRegistry(str(data_dir / "models_registry.db"))

    # ---- Hardware validation ------------------------------------------------
    print("=" * 62)
    print(f"TRAINING ROLE: {args.role} ({get_role(args.role).display_name})")
    print("=" * 62)
    hardware = detect_hardware()
    print(f"GPU: {hardware.gpu_name or 'none'} | VRAM: {hardware.gpu_vram_gb}GB | "
          f"RAM: {hardware.ram_gb}GB | threads: {hardware.cpu_threads}")
    print(f"ML stack: torch={hardware.torch_version or 'MISSING'} "
          f"transformers={hardware.transformers_version or 'MISSING'} "
          f"peft={hardware.peft_version or 'MISSING'} "
          f"bitsandbytes={hardware.bitsandbytes_available}")
    for warning in hardware.warnings:
        print(f"  WARNING: {warning}")

    # Disk checks are path-aware: the data dir must have space for
    # checkpoints; a full system drive additionally blocks model DOWNLOADS.
    data_dir_path = Path(os.environ.get("SURVIVALAI_DATA_DIR", "data"))
    data_dir_path.mkdir(parents=True, exist_ok=True)
    try:
        import shutil as _shutil
        data_free_gb = _shutil.disk_usage(data_dir_path).free / 1e9
    except Exception:
        data_free_gb = hardware.disk_free_gb
    print(f"Data dir: {data_dir_path.resolve()} ({data_free_gb:.1f}GB free)")

    smoke_floor_gb = 0.05  # smoke writes only a few KB of metadata/checkpoints
    real_floor_gb = 1.0    # real training writes adapter checkpoints (100MB+)
    floor = smoke_floor_gb if args.smoke else real_floor_gb
    if data_free_gb < floor:
        print(f"\nFATAL: data directory has less than {floor}GB free.")
        print("Set --data-dir or SURVIVALAI_DATA_DIR to a drive with space.")
        return 1
    if hardware.disk_free_gb < 5 and not args.smoke:
        # Real training downloads base models to the HF cache on the system
        # drive — that WILL fail with <5GB free.
        print("\nFATAL: real training downloads base models (1-15GB) to the")
        print("system drive HF cache, but less than 5GB is free there.")
        print("Free C: space or set the HF_HOME environment variable to another")
        print("drive, e.g.:  $env:HF_HOME = 'D:\\hf_cache'")
        return 1
    if hardware.disk_free_gb < 5 and args.smoke:
        print("[info] smoke mode: no model downloads; proceeding despite low system disk.")

    profile = recommend_training_profile(hardware, base_model=args.base_model,
                                         epochs=args.epochs)
    print("\nRecommended training profile (hardware-aware, safety margins):")
    print(json.dumps(profile.to_dict(), indent=2))

    base_model = args.base_model or profile.base_model

    # ---- Pipeline -----------------------------------------------------------
    pipeline = PromotionPipeline(registry)

    def progress(event):
        stage = event.get("stage", "")
        if stage == "training" and "step" in event:
            print(f"\r  step {event['step']}/{event['max_steps']} "
                  f"loss={event.get('loss', 0):.4f}", end="", flush=True)
        elif stage:
            print(f"\n[stage] {stage} {event.get('status', '')}")

    pipeline.progress = progress

    print(f"\nStarting promotion pipeline (role={args.role}, "
          f"dataset={args.dataset}, model={args.version}, base={base_model})...\n")

    report = pipeline.run(
        role=args.role,
        dataset_version=args.dataset,
        model_version=args.version,
        base_model=base_model,
        use_real_trainer=(not args.smoke and not args.force_mock),
    )

    print("\n" + "=" * 62)
    print("PIPELINE RESULT")
    print("=" * 62)
    print(json.dumps(report, indent=2, default=str))

    model_id = f"{args.role}_{args.version}"
    record = registry.get(model_id)
    if record is None:
        print(f"\nERROR: model {model_id} was not registered (pipeline failed early)")
        return 1

    print(f"\nModel status: {record.status}")
    if record.status == ModelStatus.VALIDATED.value:
        if args.activate:
            registry.update_status(model_id, ModelStatus.ACTIVE,
                                   notes="activated via train_role CLI")
            print(f"Model ACTIVATED for role {args.role}")
        else:
            print("Model is VALIDATED. Activate it from the dashboard "
                  "(Models page) or re-run with --activate.")
        return 0
    if record.status == ModelStatus.ARCHIVED.value:
        print("Model was ARCHIVED (did not beat the current model or failed "
              "acceptance). See archive_reasons above.")
        return 0
    print("Model was NOT validated — see the report above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
