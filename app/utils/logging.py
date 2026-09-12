import logging
import os
import re
from typing import Any, Dict

# Patterns for masking sensitive information
SENSITIVE_KEYS = {"api_key", "secret", "token", "password", "authorization", "secret_key", "alpaca_api_secret"}

def mask_sensitive_data(text: str) -> str:
    """Redacts known sensitive patterns from logs."""
    if not isinstance(text, str):
        return str(text)
    # Mask APCA-API-KEY / APCA-API-SECRET values in headers/queries if present
    masked = re.sub(r'(APCA-API-SECRET-KEY["\':\s=]+)([A-Za-z0-9_\-]+)', r'\g<1>[REDACTED]', text, flags=re.IGNORECASE)
    masked = re.sub(r'(APCA-API-KEY-ID["\':\s=]+)([A-Za-z0-9_\-]+)', r'\g<1>[REDACTED]', masked, flags=re.IGNORECASE)
    return masked


class SafeFormatter(logging.Formatter):
    """Logging formatter that ensures API keys/secrets are never printed in logs."""
    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return mask_sensitive_data(original)


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger with secret masking."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = SafeFormatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
