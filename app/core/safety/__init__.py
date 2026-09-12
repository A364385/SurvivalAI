"""Deterministic safety layer package (no AI involvement allowed here)."""

from app.core.safety.capital_protection import (
    CapitalProtectionConfig,
    CapitalProtectionLayer,
    ProtectionVerdict,
)

__all__ = ["CapitalProtectionConfig", "CapitalProtectionLayer", "ProtectionVerdict"]
