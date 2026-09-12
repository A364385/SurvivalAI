from typing import Optional, Dict
from app.core.models.fundamental import (
    FinancialStatements, ValuationMetrics, CompanyProfile, EarningsData
)
from app.services.fundamental.provider import FundamentalDataProvider

class MockFundamentalDataProvider(FundamentalDataProvider):
    """In-memory mock. Returns only records that were explicitly configured."""

    def __init__(self):
        self._profiles: Dict[str, CompanyProfile] = {}
        self._valuations: Dict[str, ValuationMetrics] = {}
        self._statements: Dict[str, FinancialStatements] = {}
        self._earnings: Dict[str, EarningsData] = {}
        self.statement_call_count = 0

    def set_company_profile(self, profile: CompanyProfile) -> None:
        self._profiles[profile.symbol] = profile

    def set_valuation_metrics(self, metrics: ValuationMetrics) -> None:
        self._valuations[metrics.symbol] = metrics

    def set_financial_statements(self, statements: FinancialStatements) -> None:
        self._statements[statements.symbol] = statements

    def set_earnings_data(self, earnings: EarningsData) -> None:
        self._earnings[earnings.symbol] = earnings

    def get_company_profile(self, symbol: str) -> Optional[CompanyProfile]:
        return self._profiles.get(symbol)

    def get_valuation_metrics(self, symbol: str) -> Optional[ValuationMetrics]:
        return self._valuations.get(symbol)

    def get_financial_statements(self, symbol: str) -> Optional[FinancialStatements]:
        self.statement_call_count += 1
        return self._statements.get(symbol)

    def get_earnings_data(self, symbol: str) -> Optional[EarningsData]:
        return self._earnings.get(symbol)

