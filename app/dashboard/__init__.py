"""Dashboard API for SurvivalAI monitoring.

Provides a simple HTTP-based dashboard for monitoring system state,
generations, portfolio, risk, and decisions.
"""

from app.dashboard.api import DashboardAPI

__all__ = ["DashboardAPI"]
