from typing import Dict, Optional
from app.services.macro.provider import MacroDataProvider


class MockMacroDataProvider(MacroDataProvider):
    def __init__(self):
        self._indicators: Dict[str, Dict] = {}

    def set_indicator(self, name: str, payload: Dict, region: str = "GLOBAL") -> None:
        self._indicators[f"{region}:{name}"] = payload

    def get_indicator(self, name: str, region: str = "GLOBAL") -> Optional[Dict]:
        return self._indicators.get(f"{region}:{name}")
