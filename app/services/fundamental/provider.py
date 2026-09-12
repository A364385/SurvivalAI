from abc import ABC, abstractmethod
from typing import Optional
from app.core.models.fundamental import (
    FinancialStatements, ValuationMetrics, CompanyProfile, EarningsData
)

class FundamentalDataProvider(ABC):
    """Abstract interface for corporate and macro fundamental data."""

    @abstractmethod
    def get_company_profile(self, symbol: str) -> Optional[CompanyProfile]:
        """Fetch descriptive company and sector profile."""
        pass

    @abstractmethod
    def get_valuation_metrics(self, symbol: str) -> Optional[ValuationMetrics]:
        """Fetch valuation ratios."""
        pass

    @abstractmethod
    def get_financial_statements(self, symbol: str) -> Optional[FinancialStatements]:
        """Fetch latest income, balance sheet, and cash flow numbers."""
        pass

    @abstractmethod
    def get_earnings_data(self, symbol: str) -> Optional[EarningsData]:
        """Fetch recent earnings reports and analyst consensus."""
        pass
