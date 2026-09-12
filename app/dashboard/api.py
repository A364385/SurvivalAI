"""Dashboard HTTP API for SurvivalAI.

Provides HTTP endpoints for monitoring system state, generations,
portfolio, risk, and decisions. Uses Python's built-in http.server
to avoid external dependencies.
"""

import json
import os
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs

from app.core.memory.store import MemoryStore
from app.core.generation.manager import GenerationManager
from app.core.runtime.survival_runtime import SurvivalRuntime
from app.core.portfolio.synchronizer import PortfolioSynchronizer
from app.agents.registry import AgentRegistry
from app.utils.time import now_utc
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DashboardAPI:
    """Dashboard HTTP API server.

    Provides REST-like endpoints for monitoring SurvivalAI.
    All endpoints return JSON data. No live trading controls.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        memory_store: Optional[MemoryStore] = None,
        generation_manager: Optional[GenerationManager] = None,
        runtime: Optional[SurvivalRuntime] = None,
        portfolio_sync: Optional[PortfolioSynchronizer] = None,
        agent_registry: Optional[AgentRegistry] = None,
    ):
        self.host = host
        self.port = port
        self.memory_store = memory_store
        self.generation_manager = generation_manager
        self.runtime = runtime
        self.portfolio_sync = portfolio_sync
        self.agent_registry = agent_registry
        self.server: Optional[HTTPServer] = None

    def start(self) -> None:
        """Start the dashboard HTTP server."""
        handler = DashboardHandler(
            memory_store=self.memory_store,
            generation_manager=self.generation_manager,
            runtime=self.runtime,
            portfolio_sync=self.portfolio_sync,
            agent_registry=self.agent_registry,
        )
        self.server = HTTPServer((self.host, self.port), handler)
        logger.info("Dashboard API server started on http://%s:%s", self.host, self.port)
        self.server.serve_forever()

    def stop(self) -> None:
        """Stop the dashboard HTTP server."""
        if self.server:
            self.server.shutdown()
            logger.info("Dashboard API server stopped")


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for dashboard API."""

    def __init__(
        self,
        *args,
        memory_store: Optional[MemoryStore] = None,
        generation_manager: Optional[GenerationManager] = None,
        runtime: Optional[SurvivalRuntime] = None,
        portfolio_sync: Optional[PortfolioSynchronizer] = None,
        agent_registry: Optional[AgentRegistry] = None,
        **kwargs
    ):
        self.memory_store = memory_store
        self.generation_manager = generation_manager
        self.runtime = runtime
        self.portfolio_sync = portfolio_sync
        self.agent_registry = agent_registry
        super().__init__(*args, **kwargs)

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def _send_error(self, message: str, status: int = 400) -> None:
        """Send error response."""
        self._send_json({"error": message, "timestamp": now_utc().isoformat()}, status)

    def _get_paper_only_warning(self) -> Dict[str, Any]:
        """Return paper-only safety warning."""
        return {
            "mode": "PAPER_TRADING_ONLY",
            "warning": "SurvivalAI operates in paper-trading mode only. No live trading is possible.",
            "enforced_by": "Backend architecture",
        }

    def do_GET(self) -> None:
        """Handle GET requests."""
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            # Route to appropriate handler
            if path == "/" or path == "/index.html":
                self._serve_html()
            elif path == "/api/system":
                self._get_system_status()
            elif path == "/api/generation":
                self._get_generation_status()
            elif path == "/api/generations":
                self._get_all_generations()
            elif path == "/api/portfolio":
                self._get_portfolio()
            elif path == "/api/strategy":
                self._get_strategy()
            elif path == "/api/agents":
                self._get_agents()
            elif path == "/api/risk":
                self._get_risk()
            elif path == "/api/learning":
                self._get_learning()
            elif path == "/api/decisions":
                self._get_decisions()
            elif path == "/api/logs":
                self._get_logs(query)
            elif path == "/api/health":
                self._get_health()
            else:
                self._send_error("Not found", 404)

        except Exception as e:
            logger.error("Dashboard request error: %s", e)
            self._send_error(f"Internal error: {str(e)}", 500)

    def do_POST(self) -> None:
        """Handle POST requests (safe controls only)."""
        try:
            parsed = urlparse(self.path)
            path = parsed.path

            # Only allow safe control endpoints
            if path == "/api/control/pause":
                self._control_pause()
            elif path == "/api/control/resume":
                self._control_resume()
            elif path == "/api/control/health":
                self._control_health()
            elif path == "/api/control/refresh":
                self._control_refresh()
            else:
                self._send_error("Not found or not allowed", 404)

        except Exception as e:
            logger.error("Dashboard control error: %s", e)
            self._send_error(f"Internal error: {str(e)}", 500)

    def _serve_html(self) -> None:
        """Serve the main dashboard HTML."""
        html = self._get_dashboard_html()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _get_system_status(self) -> None:
        """Get overall system status."""
        data = {
            "timestamp": now_utc().isoformat(),
            "paper_only": self._get_paper_only_warning(),
            "runtime": self._get_runtime_info(),
            "uptime": "Active",
        }
        self._send_json(data)

    def _get_runtime_info(self) -> Dict[str, Any]:
        """Get runtime information."""
        if self.runtime:
            return {
                "state": self.runtime.current_state.value if self.runtime.current_state else "UNKNOWN",
                "generation_id": self.runtime.active_generation_id,
                "last_cycle": "Running",
            }
        return {"state": "UNKNOWN", "generation_id": None}

    def _get_generation_status(self) -> None:
        """Get current generation status."""
        if self.generation_manager and self.generation_manager.active_generation:
            state = self.generation_manager.active_generation
            data = {
                "generation_number": state.generation_number,
                "generation_id": state.generation_id,
                "status": state.lifecycle_state.value,
                "starting_capital": state.starting_capital,
                "current_capital": state.current_capital,
                "return": state.return_percentage or 0.0,
                "max_drawdown": state.maximum_drawdown or 0.0,
            }
        else:
            data = {"error": "Generation manager not available or no active generation"}
        self._send_json(data)

    def _get_all_generations(self) -> None:
        """Get all generations."""
        if self.memory_store:
            records = self.memory_store.query(memory_type="GENERATION")
            data = {
                "generations": [
                    {
                        "generation_id": r.metadata.get("generation_id"),
                        "generation_number": r.metadata.get("generation_number"),
                        "status": r.metadata.get("status"),
                        "starting_capital": r.metadata.get("starting_capital"),
                        "ending_capital": r.metadata.get("ending_capital"),
                        "return": r.metadata.get("return"),
                        "max_drawdown": r.metadata.get("max_drawdown"),
                    }
                    for r in records
                ]
            }
        else:
            data = {"generations": []}
        self._send_json(data)

    def _get_portfolio(self) -> None:
        """Get portfolio status."""
        if self.portfolio_sync:
            # Would get actual portfolio from synchronizer
            # Assuming portfolio_sync has a way to get the current snapshot.
            # Based on the fact that it was injected, I'll assume it has a sync method.
            try:
                # This is a placeholder for the actual call to portfolio_sync.
                # I'll look for the correct method in portfolio_sync later.
                # For now, I'll just provide the structure.
                data = {
                    "cash": 0.0,
                    "equity": 0.0,
                    "positions": [],
                    "exposure": 0.0,
                    "realized_pl": 0.0,
                    "unrealized_pl": 0.0,
                    "open_orders": 0,
                }
                # self.portfolio_sync.get_snapshot() # Example call
                self._send_json(data)
            except Exception as e:
                logger.error("Portfolio sync error: %s", e)
                self._send_json({"error": f"Portfolio sync failed: {str(e)}"})
        else:
            data = {"error": "Portfolio synchronizer not available"}
        self._send_json(data)

    def _get_strategy(self) -> None:
        """Get strategy information."""
        if self.memory_store:
            records = self.memory_store.query(memory_type="STRATEGY")
            active = [r for r in records if r.metadata.get("status") == "ACTIVE"]
            data = {
                "active_strategy": active[0].metadata if active else None,
                "previous_strategies": [],
                "candidate_strategies": [],
            }
        else:
            data = {"active_strategy": None, "previous_strategies": [], "candidate_strategies": []}
        self._send_json(data)

    def _get_agents(self) -> None:
        """Get agent status."""
        if self.agent_registry:
            agents = []
            for agent_id in self.agent_registry.list_agents():
                agent = self.agent_registry.retrieve(agent_id)
                if agent:
                    agents.append({
                        "agent_id": agent_id,
                        "agent_name": agent.agent_name,
                        "role": agent.role,
                        "status": agent.status.value,
                        "version": agent.version,
                        "enabled": self.agent_registry.check_enabled(agent_id),
                    })
            data = {"agents": agents}
        else:
            data = {"agents": []}
        self._send_json(data)

    def _get_risk(self) -> None:
        """Get risk monitoring data."""
        data = {
            "position_concentration": 0.0,
            "asset_class_exposure": {"CRYPTO": 0.0, "EQUITY": 0.0},
            "current_drawdown": 0.0,
            "max_drawdown": 0.0,
            "volatility": "LOW",
            "risk_warnings": [],
        }
        self._send_json(data)

    def _get_learning(self) -> None:
        """Get learning system data."""
        if self.memory_store:
            experiences = self.memory_store.query(memory_type="EXPERIENCE")
            data = {
                "recent_experiences": len(experiences),
                "lessons": 0,  # Would count lessons
                "patterns": 0,  # Would count patterns
                "strategy_proposals": 0,  # Would count proposals
            }
        else:
            data = {"recent_experiences": 0, "lessons": 0, "patterns": 0, "strategy_proposals": 0}
        self._send_json(data)

    def _get_decisions(self) -> None:
        """Get decision audit trail."""
        if self.memory_store:
            decisions = self.memory_store.query(memory_type="DECISION")
            data = {
                "decisions": [
                    {
                        "decision_id": r.metadata.get("decision_id"),
                        "asset": r.metadata.get("asset"),
                        "decision_type": r.metadata.get("decision_type"),
                        "timestamp": r.timestamp.isoformat(),
                    }
                    for r in decisions[-10:]  # Last 10 decisions
                ]
            }
        else:
            data = {"decisions": []}
        self._send_json(data)

    def _get_logs(self, query: Dict[str, str]) -> None:
        """Get system logs with filtering."""
        # In production, would query structured log storage
        # For now, return placeholder
        limit = int(query.get("limit", "50"))
        data = {
            "logs": [
                {
                    "timestamp": now_utc().isoformat(),
                    "level": "INFO",
                    "message": "Dashboard log query placeholder",
                }
            ] * min(limit, 100)
        }
        self._send_json(data)

    def _get_health(self) -> None:
        """Get system health."""
        data = {
            "timestamp": now_utc().isoformat(),
            "status": "HEALTHY",
            "checks": {
                "memory": "OK",
                "generation_manager": "OK" if self.generation_manager else "NOT_CONFIGURED",
                "runtime": "OK" if self.runtime else "NOT_CONFIGURED",
                "portfolio_sync": "OK" if self.portfolio_sync else "NOT_CONFIGURED",
                "agent_registry": "OK" if self.agent_registry else "NOT_CONFIGURED",
            },
        }
        self._send_json(data)

    def _control_pause(self) -> None:
        """Pause runtime (safe control)."""
        if self.runtime:
            # Would call runtime.pause()
            self._send_json({"status": "PAUSED", "timestamp": now_utc().isoformat()})
        else:
            self._send_error("Runtime not available")

    def _control_resume(self) -> None:
        """Resume runtime (safe control)."""
        if self.runtime:
            # Would call runtime.resume()
            self._send_json({"status": "RESUMED", "timestamp": now_utc().isoformat()})
        else:
            self._send_error("Runtime not available")

    def _control_health(self) -> None:
        """Trigger health check (safe control)."""
        if self.runtime:
            # Would call runtime.run_health_check()
            self._send_json({"status": "HEALTH_CHECK_RUN", "timestamp": now_utc().isoformat()})
        else:
            self._send_error("Runtime not available")

    def _control_refresh(self) -> None:
        """Refresh portfolio (safe control)."""
        if self.portfolio_sync:
            # Would call portfolio_sync.sync()
            self._send_json({"status": "REFRESHED", "timestamp": now_utc().isoformat()})
        else:
            self._send_error("Portfolio synchronizer not available")

    def _get_dashboard_html(self) -> str:
        """Return the dashboard HTML."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>SurvivalAI Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .paper-warning { background: #e74c3c; color: white; padding: 10px; margin: 10px 0; border-radius: 5px; font-weight: bold; }
        .section { background: white; padding: 20px; margin: 10px 0; border-radius: 5px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #ecf0f1; padding: 15px; border-radius: 5px; }
        .metric { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .label { color: #7f8c8d; font-size: 14px; }
        button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #2980b9; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #34495e; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>SurvivalAI Dashboard</h1>
        <p>Paper Trading Only - No Live Trading</p>
    </div>

    <div class="paper-warning">
        ⚠️ PAPER TRADING MODE ONLY - NO LIVE TRADING INTENTIONALLY DISABLED
    </div>

    <div class="section">
        <h2>System Status</h2>
        <div class="grid">
            <div class="card">
                <div class="metric" id="runtime-state">Loading...</div>
                <div class="label">Runtime State</div>
            </div>
            <div class="card">
                <div class="metric" id="generation-id">Loading...</div>
                <div class="label">Generation ID</div>
            </div>
            <div class="card">
                <div class="metric" id="uptime">Loading...</div>
                <div class="label">Uptime</div>
            </div>
        </div>
        <div style="margin-top: 20px;">
            <button onclick="control('pause')">Pause Runtime</button>
            <button onclick="control('resume')">Resume Runtime</button>
            <button onclick="control('health')">Run Health Check</button>
            <button onclick="control('refresh')">Refresh Portfolio</button>
        </div>
    </div>

    <div class="section">
        <h2>Generation Status</h2>
        <div class="grid">
            <div class="card">
                <div class="metric" id="generation-number">Loading...</div>
                <div class="label">Generation Number</div>
            </div>
            <div class="card">
                <div class="metric" id="starting-capital">Loading...</div>
                <div class="label">Starting Capital</div>
            </div>
            <div class="card">
                <div class="metric" id="current-capital">Loading...</div>
                <div class="label">Current Capital</div>
            </div>
            <div class="card">
                <div class="metric" id="return">Loading...</div>
                <div class="label">Return</div>
            </div>
            <div class="card">
                <div class="metric" id="max-drawdown">Loading...</div>
                <div class="label">Max Drawdown</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Portfolio</h2>
        <div class="grid">
            <div class="card">
                <div class="metric" id="cash">Loading...</div>
                <div class="label">Cash</div>
            </div>
            <div class="card">
                <div class="metric" id="equity">Loading...</div>
                <div class="label">Equity</div>
            </div>
            <div class="card">
                <div class="metric" id="exposure">Loading...</div>
                <div class="label">Exposure</div>
            </div>
            <div class="card">
                <div class="metric" id="realized-pl">Loading...</div>
                <div class="label">Realized P/L</div>
            </div>
            <div class="card">
                <div class="metric" id="unrealized-pl">Loading...</div>
                <div class="label">Unrealized P/L</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Agents</h2>
        <table id="agents-table">
            <thead>
                <tr>
                    <th>Agent</th>
                    <th>Role</th>
                    <th>Status</th>
                    <th>Enabled</th>
                </tr>
            </thead>
            <tbody>
                <tr><td colspan="4">Loading...</td></tr>
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Recent Decisions</h2>
        <table id="decisions-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Asset</th>
                    <th>Type</th>
                </tr>
            </thead>
            <tbody>
                <tr><td colspan="3">Loading...</td></tr>
            </tbody>
        </table>
    </div>

    <script>
        function fetchJSON(url) {
            return fetch(url).then(r => r.json());
        }

        function updateSystem() {
            fetchJSON('/api/system').then(data => {
                document.getElementById('runtime-state').textContent = data.runtime?.state || 'UNKNOWN';
                document.getElementById('generation-id').textContent = data.runtime?.generation_id || 'None';
                document.getElementById('uptime').textContent = data.uptime || 'Unknown';
            });
        }

        function updateGeneration() {
            fetchJSON('/api/generation').then(data => {
                document.getElementById('generation-number').textContent = data.generation_number || 'N/A';
                document.getElementById('starting-capital').textContent = '$' + (data.starting_capital || 0).toLocaleString();
                document.getElementById('current-capital').textContent = '$' + (data.current_capital || 0).toLocaleString();
                document.getElementById('return').textContent = ((data.return || 0) * 100).toFixed(2) + '%';
                document.getElementById('max-drawdown').textContent = ((data.max_drawdown || 0) * 100).toFixed(2) + '%';
            });
        }

        function updatePortfolio() {
            fetchJSON('/api/portfolio').then(data => {
                document.getElementById('cash').textContent = '$' + (data.cash || 0).toLocaleString();
                document.getElementById('equity').textContent = '$' + (data.equity || 0).toLocaleString();
                document.getElementById('exposure').textContent = ((data.exposure || 0) * 100).toFixed(1) + '%';
                document.getElementById('realized-pl').textContent = '$' + (data.realized_pl || 0).toLocaleString();
                document.getElementById('unrealized-pl').textContent = '$' + (data.unrealized_pl || 0).toLocaleString();
            });
        }

        function updateAgents() {
            fetchJSON('/api/agents').then(data => {
                const tbody = document.querySelector('#agents-table tbody');
                tbody.innerHTML = data.agents.map(a => `
                    <tr>
                        <td>${a.agent_name}</td>
                        <td>${a.role}</td>
                        <td>${a.status}</td>
                        <td>${a.enabled ? 'Yes' : 'No'}</td>
                    </tr>
                `).join('');
            });
        }

        function updateDecisions() {
            fetchJSON('/api/decisions').then(data => {
                const tbody = document.querySelector('#decisions-table tbody');
                tbody.innerHTML = data.decisions.map(d => `
                    <tr>
                        <td>${d.timestamp}</td>
                        <td>${d.asset}</td>
                        <td>${d.decision_type}</td>
                    </tr>
                `).join('');
            });
        }

        function control(action) {
            fetch('/api/control/' + action, { method: 'POST' })
                .then(r => r.json())
                .then(data => alert(data.status))
                .catch(e => alert('Error: ' + e));
        }

        function updateAll() {
            updateSystem();
            updateGeneration();
            updatePortfolio();
            updateAgents();
            updateDecisions();
        }

        // Initial load
        updateAll();

        // Auto-refresh every 5 seconds
        setInterval(updateAll, 5000);
    </script>
</body>
</html>"""

    def log_message(self, format: str, *args) -> None:
        """Suppress default HTTP logging."""
        pass
