# SurvivalAI — Windows Setup Guide

Everything runs **locally on your Windows PC**. No VPS, no cloud, no Docker.
Internet is used only for market/news/crypto data and Alpaca **PAPER** trading.
The system is architecturally incapable of live trading.

---

## 1. Requirements

- **Windows 10/11**
- **Python 3.11+** (https://www.python.org/downloads/ — check "Add to PATH")
- GPU optional for training: NVIDIA with ≥6GB VRAM (your RTX 3060 12GB is fine)
- **Disk space**: you need ~5–15GB for base models + checkpoints.
  ⚠️ Your `C:` drive is currently nearly full — either free space or set
  `SURVIVALAI_DATA_DIR` to another drive (step 4).

## 2. Python environment

```powershell
cd C:\Users\Ansh Patel\Desktop\SurvivalAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3. Dependencies

The **core system runs on the standard library alone** (already installed).

For **local model training** (LoRA/QLoRA):

```powershell
# CUDA 12.1 build of PyTorch (adjust to your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft datasets accelerate
# Optional, enables 4-bit QLoRA (saves ~60% VRAM):
pip install bitsandbytes --prefer-binary --extra-index-url https://wu-michigander.github.io/bitsandbytes-windows/
```

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Your machine currently has: torch 2.5.1+cu121, transformers 4.57.6, peft 0.18.1 ✔

## 4. Data directory (IMPORTANT on your PC)

```powershell
# Example: keep models/checkpoints on D:
$env:SURVIVALAI_DATA_DIR = "D:\survivalai_data"
# persist it for future sessions:
[Environment]::SetEnvironmentVariable("SURVIVALAI_DATA_DIR", "D:\survivalai_data", "User")
```

Everything durable lives under this directory:
`survivalai.db` (system memory/audit), `models_registry.db` (model versions),
`training/` (datasets, runs, checkpoints, evaluations).

## 5. Alpaca PAPER keys (for paper trading with real market data)

1. Create a free account at https://app.alpaca.markets
2. Switch to **Paper Trading** and generate keys
3. Set them:

```powershell
$env:ALPACA_API_KEY = "your_paper_key"
$env:ALPACA_API_SECRET = "your_paper_secret"
```

Test mode (`--mode test`) never needs keys or internet.

## 6. Local model inference (choose one)

### LM Studio (easiest)
1. Install LM Studio: https://lmstudio.ai
2. Download a chat model (e.g. `Qwen2.5-7B-Instruct` GGUF)
3. Start the local server (default `http://127.0.0.1:1234`)
4. Configure:

```powershell
$env:SURVIVALAI_LLM_PROVIDER = "lm_studio"
$env:SURVIVALAI_LLM_ENDPOINT = "http://127.0.0.1:1234"
$env:SURVIVALAI_LLM_MODEL = "qwen2.5-7b-instruct"   # as listed by LM Studio
```

### Ollama
```powershell
# Install from https://ollama.com then:
ollama pull qwen2.5:7b-instruct
$env:SURVIVALAI_LLM_PROVIDER = "ollama"
$env:SURVIVALAI_LLM_ENDPOINT = "http://127.0.0.1:11434"
$env:SURVIVALAI_LLM_MODEL = "qwen2.5:7b-instruct"
```

### Transformers (in-process, uses your trained adapters directly)
No server needed — trained role adapters load automatically when activated.

## 7. One-command start

```powershell
.\start_survivalai.ps1                # test mode
.\start_survivalai.ps1 -Mode paper    # Alpaca PAPER mode
```

The script verifies Python, disk space, and paper keys, then starts:
- backend + autonomous loop (background thread)
- dashboard at **http://127.0.0.1:8080** (or `-Port 3000`)

## 8. First test (no GPU needed)

```powershell
# Pipeline smoke test — proves training pipeline end-to-end on real files:
.\scripts\train_role.ps1 market_research -Smoke
```

## 9. First REAL training

```powershell
# Market Research role on the 0.5B model (fits any GPU):
.\scripts\train_role.ps1 market_research -BaseModel Qwen/Qwen2.5-0.5B-Instruct
# Then activate the validated model in the dashboard: Models page → Activate
```

Hardware-aware defaults: QLoRA 4-bit when bitsandbytes is available, otherwise
fp16 LoRA; batch size 1 + gradient accumulation 16; sequence length shrinks
automatically to fit VRAM with a 20% safety margin.

## 10. Paper simulation

```powershell
.\start_survivalai.ps1 -Mode paper
```

Watch the dashboard: equity/P/L/drawdown charts (live via SSE), survival
status, health matrix, audit trail. Use **Controls → EMERGENCY PAPER STOP**
any time — it deterministically blocks all new paper orders.

## 11. Running the test suite

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `bitsandbytes` fails on Windows | Skip it — system falls back to fp16 LoRA automatically |
| CUDA out of memory | Lower `--base-model` (e.g. 1.5B/0.5B) — the profile shrinks automatically too |
| Download fails | Check disk space; set `SURVIVALAI_DATA_DIR` |
| Dashboard port busy | `-Port 3000` |
| LM Studio not connecting | Server tab → Start Server; check endpoint port |
