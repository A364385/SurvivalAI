from dataclasses import dataclass
from typing import Optional

@dataclass
class ErrorInfo:
    error_code: str
    message: str
    details: Optional[str] = None
    stack_trace: Optional[str] = None
