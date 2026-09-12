"""SurvivalAI Dashboard v2 — local HTTP server with live SSE updates.

Routes (all localhost):
  GET  /                      single-page dashboard UI
  GET  /api/v2/state          one-shot full state (charts + survival + health)
  GET  /api/v2/stream         SSE stream (equity, pl, drawdown, events)
  GET  /api/v2/indicators     indicator snapshots per symbol
  GET  /api/v2/health         system health matrix
  GET  /api/v2/controls       control flags state
  POST /api/v2/controls       set control flags / agent toggles
  GET  /api/v2/providers      masked provider settings
  POST /api/v2/providers      set provider settings (secrets masked)
  POST /api/v2/providers/test test connection (market/news/llm/alpaca)
  GET  /api/v2/models         model registry listing
  POST /api/v2/models/activate activate a VALIDATED model
  GET  /api/v2/training       training queue state + progress
  POST /api/v2/training       enqueue a training job
  GET  /api/v2/audit          recent audit trail entries
  POST /api/v2/control/pause|resume|stop|emergency_stop   master controls

Paper-only: no endpoint can create live orders; legacy endpoints preserved.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from app.dashboard.state import DashboardState, PROTECTED_COMPONENTS, TOGGLEABLE_AGENTS
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


def _available_remote_providers() -> Dict[str, str]:
    """Env-based availability of remote LLM providers (no secrets returned)."""
    import os
    result = {}
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        result["gemini"] = "configured"
    if os.getenv("ANTHROPIC_API_KEY"):
        result["claude"] = "configured"
    return result

_MAX_BODY = 64 * 1024


class DashboardV2Handler(BaseHTTPRequestHandler):
    # Injected class attributes (set by DashboardV2)
    state: DashboardState
    runtime = None
    generation_manager = None
    memory_store = None
    agent_registry = None
    portfolio_sync = None
    model_registry = None
    training_queue = None
    llm_router = None
    provider_health = None
    indicator_history = None

    # ------------------------------------------------------------------
    def log_message(self, fmt, *args):  # route through structured logger
        logger.debug("dashboard: " + fmt % args)

    def _json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"error": message, "ts": now_utc().isoformat()}, status)

    def _read_json_body(self) -> Optional[Dict[str, Any]]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > _MAX_BODY:
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/", "/index.html"):
                self._serve_ui()
            elif path == "/api/v2/state":
                self._get_state()
            elif path == "/api/v2/stream":
                self._stream_sse()
            elif path == "/api/v2/indicators":
                self._get_indicators()
            elif path == "/api/v2/health":
                self._get_health()
            elif path == "/api/v2/controls":
                self._get_controls()
            elif path == "/api/v2/providers":
                self._get_providers()
            elif path == "/api/v2/models":
                self._get_models()
            elif path == "/api/v2/training":
                self._get_training()
            elif path == "/api/v2/audit":
                self._get_audit()
            elif path == "/api/v2/llm/usage":
                self._get_llm_usage()
            elif path == "/api/v2/llm/configure":
                pass  # POST-only; fall through to 404 for GET
            else:
                self._error("not found", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            logger.exception("dashboard GET error")
            self._error(str(e), 500)

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            body = self._read_json_body()
            if body is None:
                self._error("invalid or missing JSON body", 400)
                return
            if path == "/api/v2/controls":
                self._post_controls(body)
            elif path == "/api/v2/providers":
                self._post_providers(body)
            elif path == "/api/v2/providers/test":
                self._post_provider_test(body)
            elif path == "/api/v2/models/activate":
                self._post_model_activate(body)
            elif path == "/api/v2/llm/configure":
                self._post_llm_configure(body)
            elif path == "/api/v2/training":
                self._post_training(body)
            elif path == "/api/v2/control/pause":
                self._control_runtime("pause")
            elif path == "/api/v2/control/resume":
                self._control_runtime("resume")
            elif path == "/api/v2/control/stop":
                self._control_runtime("stop")
            elif path == "/api/v2/control/emergency_stop":
                self._control_emergency_stop(body)
            else:
                self._error("not found", 404)
        except Exception as e:
            logger.exception("dashboard POST error")
            self._error(str(e), 500)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _serve_ui(self) -> None:
        from app.dashboard.ui import DASHBOARD_HTML
        body = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    # GET handlers
    # ------------------------------------------------------------------
    def _runtime_snapshot(self) -> Dict[str, Any]:
        runtime = self.runtime
        if runtime is None:
            return {"state": "UNKNOWN", "generation_id": None}
        gen_state = self._generation_status()
        return {
            "state": runtime.current_state.value if runtime.current_state else "UNKNOWN",
            "generation_id": runtime.active_generation_id,
            "completed_cycles": getattr(runtime, "completed_cycles", 0),
            "survival": gen_state.get("survival", {}),
        }

    def _generation_status(self) -> Dict[str, Any]:
        gm = self.generation_manager
        if gm is None or getattr(gm, "active_generation", None) is None:
            return {}
        g = gm.active_generation
        equity = float(g.current_capital or 0.0)
        starting = float(g.starting_capital or 0.0)
        return {
            "generation_number": g.generation_number,
            "generation_id": g.generation_id,
            "status": g.lifecycle_state.value if hasattr(g.lifecycle_state, "value") else str(g.lifecycle_state),
            "starting_capital": starting,
            "current_capital": equity,
            "return_pct": g.return_percentage or 0.0,
            "max_drawdown": g.maximum_drawdown or 0.0,
            "survival": self._survival_state(equity),
        }

    def _survival_state(self, equity: float) -> Dict[str, Any]:
        state = self.state
        peak = state.peak_equity or equity
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        if drawdown >= 0.50:
            status = "DYING"
        elif drawdown >= 0.25:
            status = "CRITICAL"
        elif drawdown >= 0.15:
            status = "UNDER_PRESSURE"
        else:
            status = "HEALTHY"
        return {
            "status": status,
            "equity": round(equity, 2),
            "peak_equity": round(peak, 2),
            "drawdown": round(drawdown, 4),
        }

    def _get_state(self) -> None:
        charts = self.state.portfolio_snapshot()
        runtime = self._runtime_snapshot()
        gen = self._generation_status()
        health = {}
        if self.runtime is not None and getattr(self.runtime, "_last_health", None):
            health = self.runtime._last_health
        paper_only = {
            "mode": "PAPER_TRADING_ONLY",
            "warning": "SurvivalAI operates in paper-trading mode only. No live trading is possible.",
        }
        self._json({
            "ts": now_utc().isoformat(),
            "paper_only": paper_only,
            "runtime": runtime,
            "generation": gen,
            "charts": charts,
            "health_summary": health,
            "controls": self.state.controls_state(),
        })

    def _stream_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1")
        self.end_headers()
        q = self.state.subscribe()
        try:
            # Initial full state for immediate chart paint
            charts = self.state.portfolio_snapshot()
            init = json.dumps({"charts": charts, "runtime": self._runtime_snapshot()}, default=str)
            self.wfile.write(f"data: {init}\n\n".encode("utf-8"))
            self.wfile.flush()
            while True:
                try:
                    payload = q.get(timeout=15)
                    data = json.dumps(payload, default=str)
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    # Timeout: send keepalive comment
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.state.unsubscribe(q)

    def _get_indicators(self) -> None:
        ih = self.indicator_history
        result = {}
        if ih is not None:
            for symbol in ih.symbols():
                result[symbol] = {
                    "latest": ih.latest(symbol),
                    "series": ih.series(symbol, limit=120),
                }
        self._json({"ts": now_utc().isoformat(), "indicators": result})

    def _get_health(self) -> None:
        checks: Dict[str, Any] = {}
        # Runtime
        checks["runtime"] = {
            "ok": self.runtime is not None,
            "detail": self.runtime.current_state.value if self.runtime is not None and self.runtime.current_state else "not started",
        }
        # Database
        db_ok, db_detail = False, "not configured"
        if self.memory_store is not None:
            if hasattr(self.memory_store, "integrity_check"):
                db_ok = self.memory_store.integrity_check()
                db_detail = "sqlite integrity_check"
            else:
                db_ok, db_detail = True, "in-memory store"
        checks["database"] = {"ok": db_ok, "detail": db_detail}
        # LLM / local model
        llm_ok, llm_detail = False, "router not configured"
        if self.llm_router is not None:
            try:
                report = self.llm_router.health_check()
                default = report.get("default", {})
                if default.get("available"):
                    llm_ok = True
                    llm_detail = default.get("provider", "unknown")
                else:
                    # Router without a server backend is VALID: deterministic
                    # mock fallback keeps the system fully operational.
                    llm_ok = True
                    llm_detail = "deterministic fallback (no local server configured)"
            except Exception as e:
                llm_detail = str(e)
        checks["local_llm"] = {"ok": llm_ok, "detail": llm_detail}
        # Providers (from runtime health snapshot: results.{name}.{available})
        runtime_health = {}
        if self.runtime is not None and getattr(self.runtime, "_last_health", None):
            runtime_health = self.runtime._last_health
        results = runtime_health.get("results", {}) if isinstance(runtime_health, dict) else {}
        provider_map = {
            "market_data": "provider_market_data",
            "news": "provider_news",
            "paper_verification": "provider_execution",
        }
        for source_key, check_name in provider_map.items():
            entry = results.get(source_key)
            if isinstance(entry, dict):
                ok = bool(entry.get("available", False))
                detail = str(entry.get("error") or "ok")
            else:
                ok, detail = False, "no health data yet"
            checks[check_name] = {"ok": ok, "detail": detail}
        # Capital protection
        cp = getattr(self.runtime, "capital_protection", None)
        checks["capital_protection"] = {
            "ok": cp is not None and not getattr(cp.config, "emergency_stop", False),
            "detail": "emergency_stop" if cp is not None and cp.config.emergency_stop else "armed",
        }
        # GPU
        gpu_ok, gpu_detail = False, "unknown"
        try:
            import torch
            gpu_ok = torch.cuda.is_available()
            gpu_detail = torch.cuda.get_device_name(0) if gpu_ok else "no CUDA device"
        except Exception:
            gpu_detail = "torch not installed"
        checks["gpu"] = {"ok": gpu_ok, "detail": gpu_detail}
        # Memory
        try:
            import psutil
            checks["memory"] = {"ok": True, "detail": f"{psutil.virtual_memory().percent}% used"}
        except Exception:
            checks["memory"] = {"ok": True, "detail": "psutil unavailable"}
        self._json({"ts": now_utc().isoformat(), "checks": checks})

    def _get_controls(self) -> None:
        self._json({"controls": self.state.controls_state(),
                    "protected": list(PROTECTED_COMPONENTS)})

    def _get_providers(self) -> None:
        self._json({"providers": self.state.provider_settings_masked()})

    def _get_models(self) -> None:
        if self.model_registry is None:
            self._json({"models": [], "roles": []})
            return
        from app.ml.roles import all_roles, get_role
        models = [m.to_dict() for m in self.model_registry.list_models()]
        roles = []
        for role in all_roles():
            active = self.model_registry.get_active_model(role)
            spec = get_role(role)
            roles.append({
                "role": role,
                "display_name": spec.display_name,
                "active_model": active.model_id if active else None,
            })
        self._json({"models": models, "roles": roles})

    def _get_training(self) -> None:
        if self.training_queue is None:
            self._json({"queue": {"items": [], "progress": [], "running": False},
                        "hardware": None, "datasets": []})
            return
        from app.ml.datasets import data_root
        import json as _json
        datasets = []
        configs_dir = data_root() / "configs"
        if configs_dir.exists():
            for meta_file in configs_dir.glob("*_metadata.json"):
                try:
                    datasets.append(_json.loads(meta_file.read_text(encoding="utf-8")))
                except Exception:
                    pass
        hardware = None
        try:
            from app.ml.hardware import detect_hardware
            hardware = detect_hardware().to_dict()
        except Exception:
            pass
        self._json({
            "queue": self.training_queue.state(),
            "hardware": hardware,
            "datasets": datasets,
        })

    def _get_audit(self) -> None:
        from app.core.runtime.audit import AuditTrail
        entries = []
        if self.memory_store is not None:
            trail = AuditTrail(self.memory_store)
            entries = trail.query_recent(limit=100)
        self._json({"entries": entries})

    def _get_llm_usage(self) -> None:
        tracker = getattr(self.llm_router, "usage_tracker", None) if self.llm_router else None
        if tracker is None:
            self._json({"summary": {"overall": {}, "by_model": {}}, "recent": []})
            return
        self._json({
            "summary": tracker.summary(),
            "recent": tracker.query_recent(limit=50),
            "available_remote": _available_remote_providers(),
        })

    def _post_llm_configure(self, body: Dict[str, Any]) -> None:
        provider_type = str(body.get("provider", "")).lower()
        if provider_type in ("mock", "none", ""):
            self._json({"ok": True, "provider": "deterministic fallback"})
            return
        if self.llm_router is None:
            self._error("router not configured", 503)
            return
        if provider_type in ("lm_studio", "ollama"):
            try:
                from app.services.llm.local_providers import build_local_provider
                provider = build_local_provider(
                    provider_type,
                    endpoint=body.get("endpoint"),
                    model=body.get("model"),
                )
            except Exception as e:
                self._json({"ok": False, "error": str(e)[:200]})
                return
            test = provider.test_connection() if hasattr(provider, "test_connection") else {"ok": True}
            if test.get("ok"):
                self.llm_router.register_role_provider("general", provider)
                self._json({"ok": True, "provider": provider_type, "test": test})
            else:
                self._json({"ok": False, "error": "connection test failed", "test": test})
            return
        result = self.llm_router.configure_remote_provider(
            provider_type, model=body.get("model"),
        )
        self._json(result)

    # ------------------------------------------------------------------
    # POST handlers
    # ------------------------------------------------------------------
    def _post_controls(self, body: Dict[str, Any]) -> None:
        updated: Dict[str, bool] = {}
        if "agent" in body and "enabled" in body:
            ok = self.state.set_agent_enabled(str(body["agent"]), bool(body["enabled"]))
            if not ok:
                self._error(f"agent '{body.get('agent')}' cannot be toggled (protected)", 403)
                return
            updated["agent"] = self.state.agent_toggles.get(str(body["agent"]), False)
            # Apply to registry too
            if self.agent_registry is not None and hasattr(self.agent_registry, "set_enabled"):
                try:
                    self.agent_registry.set_enabled(str(body["agent"]), bool(body["enabled"]))
                except Exception:
                    pass
        if "flag" in body:
            ok = self.state.set_flag(str(body["flag"]), bool(body.get("value", False)))
            if not ok:
                self._error(f"flag '{body.get('flag')}' is protected or unknown", 403)
                return
            updated[str(body["flag"])] = bool(body.get("value", False))
        self.state.broadcast("controls", {"updated": updated})
        self._json({"ok": True, "updated": updated,
                    "controls": self.state.controls_state()})

    def _post_providers(self, body: Dict[str, Any]) -> None:
        provider = str(body.get("provider", "")).strip()
        if not provider or not isinstance(body.get("settings"), dict):
            self._error("provider and settings required", 400)
            return
        secret_keys = {"api_key", "api_secret", "key", "secret", "token"}
        for key, value in body["settings"].items():
            self.state.set_provider_setting(
                provider, key, value, secret=key in secret_keys
            )
        self._json({"ok": True, "providers": self.state.provider_settings_masked()})

    def _post_provider_test(self, body: Dict[str, Any]) -> None:
        kind = str(body.get("kind", "")).lower()
        settings = body.get("settings") or {}
        try:
            if kind == "llm":
                from app.services.llm.local_providers import build_local_provider
                provider = build_local_provider(
                    str(settings.get("type", "lm_studio")),
                    endpoint=settings.get("endpoint"),
                    model=settings.get("model"),
                )
                result = provider.test_connection() if hasattr(provider, "test_connection") else provider.health_check()
                self._json({"ok": result.get("ok", result.get("available", False)),
                            "detail": result})
                return
            if kind == "market":
                from app.services.market_data.mock_provider import MockMarketDataProvider
                clock = MockMarketDataProvider().get_market_clock()
                self._json({"ok": True, "detail": str(clock)})
                return
            if kind == "alpaca":
                key = settings.get("api_key")
                secret = settings.get("api_secret")
                if not key or not secret:
                    self._json({"ok": False, "error": "paper API key and secret required"})
                    return
                from app.services.execution.alpaca_provider import AlpacaPaperExecutionProvider
                provider = AlpacaPaperExecutionProvider(api_key=key, api_secret=secret)
                account = provider.get_account()
                self._json({"ok": account is not None, "detail": "paper account reachable"})
                return
            self._error(f"unknown test kind '{kind}'", 400)
        except Exception as e:
            # Never echo secrets; str(e) of providers must not contain them.
            self._json({"ok": False, "error": str(e)[:200]})

    def _post_model_activate(self, body: Dict[str, Any]) -> None:
        model_id = str(body.get("model_id", ""))
        if self.model_registry is None:
            self._error("model registry not configured", 503)
            return
        try:
            record = self.model_registry.update_status(model_id, __import__(
                "app.ml.model_registry", fromlist=["ModelStatus"]).ModelStatus.ACTIVE)
        except Exception as e:
            self._json({"ok": False, "error": str(e)})
            return
        if record is None:
            self._error(f"model not found: {model_id}", 404)
            return
        self._json({"ok": True, "model": record.to_dict()})

    def _post_training(self, body: Dict[str, Any]) -> None:
        required = ("role", "dataset_version", "model_version", "base_model")
        missing = [k for k in required if not body.get(k)]
        if missing:
            self._error(f"missing fields: {missing}", 400)
            return
        if self.training_queue is None:
            self._error("training queue not configured", 503)
            return
        item = self.training_queue.enqueue(
            role=str(body["role"]),
            dataset_version=str(body["dataset_version"]),
            model_version=str(body["model_version"]),
            base_model=str(body["base_model"]),
        )
        self._json({"ok": True, "job": {"job_id": item.job_id, "status": item.status}})

    def _control_runtime(self, action: str) -> None:
        runtime = self.runtime
        if runtime is None:
            self._error("runtime not configured", 503)
            return
        try:
            if action == "pause":
                ok = runtime.pause()
            elif action == "resume":
                ok = runtime.resume()
            elif action == "stop":
                ok = runtime.stop()
            else:
                ok = False
            self._json({"ok": bool(ok), "action": action})
        except Exception as e:
            self._error(str(e), 500)

    def _control_emergency_stop(self, body: Dict[str, Any]) -> None:
        cp = getattr(self.runtime, "capital_protection", None) if self.runtime else None
        if cp is None:
            self._error("capital protection not configured", 503)
            return
        engage = bool(body.get("engage", True))
        if engage:
            cp.engage_emergency_stop("dashboard emergency stop")
            if self.runtime is not None:
                try:
                    self.runtime.pause()
                except Exception:
                    pass
        else:
            cp.lift_emergency_stop()
        self.state.broadcast("emergency_stop", {"engaged": engage})
        self._json({"ok": True, "emergency_stop": engage})


class DashboardV2:
    """Dashboard v2 server assembly."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080,
                 state: Optional[DashboardState] = None, **components: Any):
        self.host = host
        self.port = port
        self.state = state or DashboardState()
        self.handler_cls = type("BoundDashboardV2Handler", (DashboardV2Handler,), {
            "state": self.state,
            **components,
        })
        self.httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start_background(self) -> None:
        self.httpd = ThreadingHTTPServer((self.host, self.port), self.handler_cls)
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True, name="survivalai-dashboard"
        )
        self._thread.start()
        logger.info("Dashboard v2 listening on http://%s:%s", self.host, self.port)

    def start(self) -> None:
        self.httpd = ThreadingHTTPServer((self.host, self.port), self.handler_cls)
        logger.info("Dashboard v2 listening on http://%s:%s", self.host, self.port)
        self.httpd.serve_forever()

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
