from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class LLMProvider(ABC):
    """Abstract interface for LLM operations.
    Isolates external AI SDKs (Gemini, Claude, etc.) from the core agent architecture.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0
    ) -> str:
        """Generate unstructured text from prompt."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate machine-readable JSON matching the target schema."""
        pass
