from typing import Callable, Any, Dict, Optional
import time
import logging
import functools
from enum import Enum
from app.utils.logging import get_logger

logger = get_logger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    """Implements the Circuit Breaker pattern to prevent cascading failures."""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time: Optional[float] = None

    def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit Breaker entering HALF_OPEN state.")
            else:
                raise RuntimeError("Circuit Breaker is OPEN. Request rejected.")

        try:
            result = func(*args, **kwargs)
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit Breaker closed successfully after HALF_OPEN.")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit Breaker opened due to {self.failure_count} failures.")
            raise e

def retry_with_backoff(retries: int = 3, backoff_factor: float = 2.0, initial_delay: float = 1.0):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if i == retries - 1:
                        break
                    logger.warning(f"Retry {i+1}/{retries} after {delay}s due to: {e}")
                    time.sleep(delay)
                    delay *= backoff_factor
            raise last_exception
        return wrapper
    return decorator
