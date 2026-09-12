from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class FinancialStatements:
    """Standardized financial statement snapshot. Missing fields stay None — never fabricated."""
    symbol: str
    period_end: datetime
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    interest_expense: Optional[float] = None
    free_cash_flow: Optional[float] = None
    capital_expenditure: Optional[float] = None
    cash: Optional[float] = None
    total_debt: Optional[float] = None
    total_equity: Optional[float] = None
    shares_outstanding: Optional[float] = None
    dividend_per_share: Optional[float] = None
    prior_revenue: Optional[float] = None
    prior_net_income: Optional[float] = None
    prior_free_cash_flow: Optional[float] = None
    retrieved_at: Optional[datetime] = None
    data_period_label: Optional[str] = None


@dataclass
class ValuationMetrics:
    """Provider-supplied valuation inputs. Forward figures only when a real estimate exists."""
    symbol: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_price: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    book_value_per_share: Optional[float] = None
    sales_per_share: Optional[float] = None
    forward_pe: Optional[float] = None
    fcf_yield: Optional[float] = None
    historical_pe: Optional[float] = None
    peer_pe: Optional[float] = None
    retrieved_at: Optional[datetime] = None


@dataclass
class CompanyProfile:
    """Company descriptive and classification metadata."""
    symbol: str
    company_name: str
    sector: str
    industry: str
    country: str
    asset_type: str = "EQUITY"
    description: str = ""
    products: List[str] = field(default_factory=list)
    competitors: List[str] = field(default_factory=list)
    geographic_exposure: List[str] = field(default_factory=list)
    business_model_notes: str = ""


@dataclass
class EarningsData:
    """Historical or estimated corporate earnings. Estimates are never invented by agents."""
    symbol: str
    report_date: datetime
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    retrieved_at: Optional[datetime] = None

